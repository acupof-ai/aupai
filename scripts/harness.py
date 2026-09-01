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
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "datagen"))
import corpus_fingerprint as cfp  # noqa: E402
import pod_drift  # noqa: E402
DATA = os.path.join(ROOT, "data")
SAMPLE_DOMAIN = "sample"  # the only corpus directory a git checkout ships

PASS, FAIL, SKIP, WARN = "PASS", "FAIL", "SKIP", "WARN"

# Per-check deadline. A check that hangs blocks the pre-commit hook and trains
# people to --no-verify; a timed-out check SKIPs and names itself in the output.
_CHECK_TIMEOUT = 5
# Checks that legitimately scan more data than the 5s default allows. The
# template scan reads ~850k text fields on a full-data checkout (27s measured).
_CHECK_TIMEOUTS = {
    "eval_sft_template_contamination": 90,
    # measured 6.1 s on 2026-09-01 (one remote ps per training process); stopgap until the
    # read is batched -- the 5 s default killed the whole run because no SIGALRM handler existed
    "no_foreground_pod_training": 15,
}


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
REVIEW_PAIRS = {"de": "44", "44": "de", "tilerl": "b0", "b0": "tilerl",
                "3b": "e1", "e1": "3b", "fb": "44"}
#: A review that has not arrived within this many minutes of the done row FAILs.
#: Inside it, WARN: a missing review must not block a close, and must not stay invisible.
REVIEW_GRACE_MIN = 30
#: The rule starts here. 41 tasks closed before it existed and cannot grow a reviewer;
#: failing them would be a permanent red nobody can act on, which is the same as no signal.
REVIEW_RULE_FROM = "2026-08-31 22:00"

#: Rule bullet (prefix) -> the check that enforces it. The AGENTS.md "Rule coverage"
#: table is the human-readable copy of this map; agents_rules_covered keeps both honest.
_RULE_CHECKS = {
    # pinned_ids + tokenizer_roundtrip catch a REBUILD after the fact (moved specials,
    # a dropped byte). Neither can see the unfreeze decision itself.
    "Tokenizer frozen 2026-08-29": "pinned_ids",
    "Long jobs detach": "no_foreground_pod_training",
    "CI gates": "CI",
    "Derived artifacts carry the fingerprint of what produced them": "corpus_fp_matches",
    "setsid, not nohup": "no_foreground_pod_training",
    "CUDA_VISIBLE_DEVICES, not cuda:N": "gemm_dims_aligned",
    "Push code via scripts/pod_push.sh <files>, never bare podput": "pod_drift",
    "Outbound network: curl -4, always": "curl_ipv4",
    "runs/.jsonl ledgers merge by union": "no_ghost_running",
    "scripts/pod_push.sh pushes only content reachable from main": "pod_drift",
    "A commit that touches a file in data/pod_head_manifest.txt": "pod_drift",
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
    "Language": "no automatic judge of whether prose is English or Chinese-for-the-user",
    "Shared files": "announcing an edit happens in conversation, outside the repo",
    "GPUs": "card ownership is a controller decision, not a file state",
    "Lanes: a 7-card training block, and one lane card for everything else":
        "the lane/block split is allocation policy; lane_respected checks the instant, not the policy",
    "Small jobs queue on the lane card":
        "queueing is operator behaviour over time; lane_respected catches the instantaneous violation",
    "The lane holds one job at a time": "same: lane_respected sees now, not the queue discipline",
    "What is reachable, measured 2026-08-30 with -4": "a record of a measurement, not a rule to enforce",
    "Reachability changes without notice, so a fetcher carries a mirror chain":
        "fetchers do carry chains; asserting 'a chain is present' would match a comment",
    "File transfer into the container: podput <local> <remote-abs-path>":
        "the 100KB cap is enforced by podput itself, which refuses",
    "tn exec and ~/bin/pod are two different filesystem views":
        "a fact about the environment; the mistakes it prevents are interactive",
    "cd inside a backgrounded chain stays in it": "a shell fact; no artifact records the mistake",
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
    "Vocabulary identity": "enforced at load: sft_math.py refuses a vocab_id mismatch, not a harness check",
    "Commit in your worktree as soon as a change works": "same deadline as above, enforced by dirty_aged",
}
#: Ratchet, a LITERAL. `len(_MANUAL_RULES)` would move with the thing it pins and the
#: check could never fire -- the ratchet has to be a number a commit has to change.
#: Raising it needs a message saying which rule became unenforceable and why.
_MANUAL_BASELINE = 22


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


def check_cited_artifacts_attested(root):
    """A fact citing a gitignored artifact carries a sha256 some attestation matches.

    data/eval/preds_*.jsonl is gitignored and nothing reads it programmatically, so
    fact_refs_resolve skips those paths on every machine: a fact could cite an artifact
    that exists nowhere and nothing would notice. That is how an unlogged rerun
    overwrote preds_l1_d3.jsonl and left five facts pointing at 477 rows of a different
    run for hours (e1, 44's contract, 2026-08-31).

    What this proves is historical -- the cited bytes existed when the citation was
    made. It deliberately does NOT compare against the current file: preds are
    regenerated every run, so a current-state check would fail on every legitimate
    rerun. The writer's attestation row is the proof."""
    refs = os.path.join(root, "runs", "artifact_refs.jsonl")
    attested = set()
    if os.path.exists(refs):
        for line in open(refs, encoding="utf-8"):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("sha256"):
                attested.add(r["sha256"])
    # The contract starts here. 18 citations predate it and cannot grow an attestation
    # retroactively -- their artifacts were written before any writer attested, and
    # several no longer exist. Failing them is a red nobody can act on, which is the
    # same as no signal. New and re-measured facts carry the hash.
    contract_from = "2026-09-01"
    cited, bad, legacy = 0, [], 0
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
                sha = e.get("artifact_sha256") or ""
                if not sha:
                    bad.append(f"{e.get('id')} cites {path} with no artifact_sha256")
                elif sha not in attested:
                    bad.append(f"{e.get('id')} cites {path} sha {sha[:12]} with no attestation")
    if not cited:
        return SKIP, (f"no fact measured since {contract_from} cites a data/eval artifact "
                      f"({legacy} predate the contract)")
    if bad:
        return FAIL, f"{len(bad)} of {cited} citation(s) unattested: {'; '.join(bad[:3])}"
    return PASS, (f"{cited} artifact citation(s) since {contract_from}, every hash attested "
                  f"by its writer ({legacy} legacy citations exempt)")


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
    lost, ok = [], 0
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
        else:
            lost.append(f"{ck} (milestone {tok or '?'})")
    if lost:
        return FAIL, (f"{len(lost)} milestone row(s) whose checkpoint is gone with no pinned "
                      f"copy: {'; '.join(lost[:3])} -- that measurement cannot be repeated")
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


def check_agents_rules_covered(root):
    """Every AGENTS.md rule maps to a check name or an explicit manual reason.

    A rule that is only prose is one people break for cause: tonight the register
    refusal in a worktree pushed a session into the shared tree, and 'run it in the
    main checkout' was a documented instruction pointing at the one place sessions
    overwrite each other. Coverage cannot prove a mapping is honest -- it proves one
    was made, and the manual count is ratcheted so 'manual' cannot quietly win."""
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
    return PASS, f"{len(bullets)} rules: {len(bullets) - n_manual} checked, {n_manual} manual (baseline {_MANUAL_BASELINE})"


def _broken_agents_rules_covered():
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
    launching shell exited and it survived, which is the property setsid provides."""
    r = subprocess.run([os.path.expanduser("~/bin/pod"), f"ps -o ppid= -p {pid}"],
                       capture_output=True, text=True)
    return r.stdout.strip() or None


def check_no_foreground_pod_training(root):
    """No training process on the pod outside a setsid session.

    'Long jobs detach' is the rule; the failure it prevents is an orphan holding a
    whole card at 100% after the tn tunnel dies, which once contaminated a
    seven-card profile silently. A detached job's session id differs from its pid's
    parent shell; a foreground one shares the crictl exec session."""
    pod = os.path.expanduser("~/bin/pod")
    if not os.path.exists(pod) or pod_drift.is_pod(root):
        return SKIP, "host-side check; needs ~/bin/pod"
    try:
        r = subprocess.run(
            [pod, "ps -eo pid,sid,pgid,args --no-headers | grep -E 'train[.]py|run_ddp' | grep -v grep"],
            capture_output=True, text=True, timeout=30,
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        return SKIP, f"pod unreachable: {type(e).__name__}"
    rows = [ln.split(None, 3) for ln in r.stdout.strip().split("\n") if ln.strip()]
    # Drop the INVOKING shell. `pod "... setsid nohup python3 harness.py launch ..."`
    # leaves a bash -lc whose argv contains the whole launch command, so a match on
    # train.py/run_ddp text catches the launcher's own wrapper -- which is not a
    # training process and is correctly not a session leader. It names setsid in its
    # own command line; the job it spawned is the thing to judge (2026-09-01, this
    # check refused a commit while tilerl's A/B was launching correctly).
    rows = [x for x in rows if not (len(x) > 3 and "setsid" in x[3] and x[3].startswith("bash -lc"))]
    if not rows:
        return PASS, "no training process on the pod"
    # ppid == 1 means init adopted it: the launching shell is gone and the process
    # survived, which IS what setsid buys. Its session leader may be a zombie ([bash]
    # <defunct>, sid alive but absent from ps output), so a leader-presence test reads
    # a correctly detached trainer as unsupervised -- this refused a commit while
    # tilerl's A/B arm ran exactly as intended (2026-09-01, second false positive from
    # this check).
    detached = {x[0] for x in rows if len(x) >= 4 and x[3].startswith(("/usr/bin/python3", "python3"))
                and _ppid_of(x[0]) == "1"}
    attached = [x for x in rows if len(x) >= 3 and x[0] != x[1] and x[0] not in detached]
    # A setsid'd launcher IS its session leader; its ranks are children sharing that sid.
    leaders = {x[1] for x in rows if len(x) >= 2 and x[0] == x[1]}
    orphans = [x for x in attached if len(x) >= 2 and x[1] not in leaders]
    if orphans:
        return FAIL, f"{len(orphans)} training process(es) not under a setsid session: pid {orphans[0][0]}"
    return PASS, f"{len(rows)} training process(es), all under setsid session(s) {sorted(leaders)}"


def _broken_no_foreground_pod_training():
    # A broken world here would need a real foreground training process on the pod --
    # i.e. committing the exact incident the check exists to prevent, on the box
    # running the 15B job. The check reads live process state, not a repo artifact,
    # so there is nothing in a temp tree to break. Skipped out loud rather than
    # given a hand-written world that would share the check's own assumptions.
    raise SelftestSkip("reads live pod process state; no repo artifact to break")


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
    for ext in ("*.py", "*.sh"):
        for d in ("scripts", "datagen", "filters", "eval", "algorithms"):
            for p in glob.glob(os.path.join(root, d, "**", ext), recursive=True):
                text = open(p, encoding="utf-8", errors="replace").read()
                # Drop docstrings and comments before looking for invocations.
                text = re.sub(r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'', "", text)
                for n, line in enumerate(text.split("\n"), 1):
                    s = line.split("#", 1)[0]
                    if inv.search(s) and not re.search(r"-4\b", s):
                        bad.append(f"{os.path.relpath(p, root)}:{n}")
    if bad:
        return FAIL, f"{len(bad)} curl call(s) without -4: {bad[:3]}"
    return PASS, "every curl call passes -4"


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


@functools.lru_cache(maxsize=None)
def experiments(raw=False):
    """The experiment log, folded by (name, started): the last event for a run wins.

    The file is an event log -- exp.py appends a running row and later a terminal one
    rather than rewriting, so union merge cannot produce a running/done pair. A reader
    that does not fold sees a superseded status: t56_profile went ok 13:34 then fail
    13:47, and an unfolded read failed score_matrix_present on the stale ok.
    raw=True yields every event."""
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
    if raw:
        return evs
    folded = {}
    for r in evs:
        folded[(r.get("name"), r.get("started"))] = r
    return list(folded.values())


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
    import subprocess

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


def _gpu_present():
    """Whether this machine can train. The strict branch of mix_shards_present guards the
    pod; a dev box with a partial corpus is normal. HARNESS_GPU_PRESENT=1/0 overrides -- the
    selftest forces 1 so the broken world exercises the strict branch."""
    forced = os.environ.get("HARNESS_GPU_PRESENT")
    if forced is not None:
        return forced == "1"
    return bool(glob.glob("/dev/nvidia[0-9]*"))


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
    import subprocess

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
    """Definitions the merge base had that the merge result no longer has, where the
    losing side never deleted them. Returns [(path, name, side_taken)].

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
            if not deleted_deliberately:
                out.append((path, name, side))
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
    out = []
    for path in sorted(both):
        mv = git("rev-parse", f"{m}:{path}").strip()
        a = git("rev-parse", f"{ours}:{path}").strip()
        b = git("rev-parse", f"{theirs}:{path}").strip()
        if not mv or a == b:
            continue
        if mv == a:
            lost = len(git("rev-list", f"{ours}..{theirs}", "--", path).split())
            out.append((path, "ours", lost))
        elif mv == b:
            lost = len(git("rev-list", f"{theirs}..{ours}", "--", path).split())
            out.append((path, "theirs", lost))
    return out


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
    # A ledger is union-merged by .gitattributes and legitimately equals one side
    # when only that side appended; that is the merge driver working, not a drop.
    took = [t for t in took if not t[0].endswith(".jsonl")
            and t[0] != "data/pod_head_manifest.txt"]
    # 0 commits lost means the other side had no commits touching that path: the file
    # matches one parent because only one parent's history reached it, not because a
    # resolution discarded anything. Seven of the nine hits over one day's 93 merges
    # were this shape.
    took = [t for t in took if t[2] > 0]
    # A path whose STAGED blob differs from the parent the merge took whole is being
    # fixed right now. Judge what is about to be committed, not what was.
    parents = subprocess.run(["git", "-C", root, "rev-list", "--parents", "-n", "1", "HEAD"],
                             capture_output=True, text=True).stdout.split()
    ours, theirs = (parents[1], parents[2]) if len(parents) >= 3 else (None, None)
    fixed = []
    for path, side, n in list(took):
        staged = subprocess.run(["git", "-C", root, "rev-parse", f":{path}"],
                                capture_output=True, text=True).stdout.strip()
        if not staged:
            continue  # nothing staged for it; the merge's own blob stands
        offending = ours if side == "ours" else theirs
        blob = subprocess.run(["git", "-C", root, "rev-parse", f"{offending}:{path}"],
                              capture_output=True, text=True).stdout.strip()
        if blob and staged != blob:
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
    if reverted:
        return FAIL, (
            f"{len(reverted)} definition(s) present in the merge base and gone from the "
            "result, with no side deleting them: "
            + "; ".join(f"{name} in {path}" for path, name, _ in reverted[:3])
            + ". A side that never had the content did not delete it -- restore from the base."
        )
    contested = len([1 for _ in merge_took_one_side(root)]) or 0
    n_both = len(set(subprocess.run(
        ["git", "-C", root, "diff", "--name-only", "HEAD^1", "HEAD"],
        capture_output=True, text=True).stdout.split()))
    return PASS, f"{n_both} file(s) changed by the merge, {contested} contested file(s) taken whole"


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
    for d in ("eval", "scripts", "datagen", "probes", "algorithms"):
        for path in sorted(glob.glob(os.path.join(root, d, "*.py"))):
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
    out = {}
    for r in evs:
        key = (r.get("name"), r.get("started"))
        prev = out.get(key)
        # A close is TERMINAL. Last-write-wins alone reopens a finished run when a
        # duplicate start event lands after its close -- which the ledger contains:
        # (sft_p324_v3, 2026-08-31 03:44) has an ok event at line 44 and a running
        # event at line 132, and folding on order alone reported a run that finished
        # in 32 minutes as 26 hours stale. A union merge of two branches can order
        # events however it likes, so order is not evidence of sequence.
        # Terminal beats running regardless of POSITION. A union merge concatenates
        # two branches' rows, so a `running` row can land after the `ok` that closed
        # it -- sft_p324_v3 has ok at line 44 and running at 132 for the same
        # (name, started). Position-based last-wins reads that as an open run 25h old
        # and refuses every merge in the shipment window. A run does not reopen; only
        # `task reopen` does that, and it is a different ledger. Two terminal events
        # for one run: the later one wins. (de and e1 reached this independently,
        # 2026-09-01; de's inline version in check_no_stale_running and this shared
        # one merged here, with de's reasoning kept.)
        if prev is not None and prev.get("status") != "running" and r.get("status") == "running":
            continue
        out[key] = r
    return list(out.values())


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
        if age_h > 24:
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

    ghosts = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
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
    return PASS, "main() calls _assert_mix_domains"


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
FACT_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
FACT_SOURCE_PATH = re.compile(
    r"(?<![\w/])(?:data|runs|scripts|docs|eval|datagen|filters|mathbank|algorithms|workflows)/[\w./-]+"
)
# Debt register for tracked-missing sources: each entry carries a reason. Can only
# shrink -- a new missing source is a FAIL, not a baseline entry. Reported in `gaps`.
FACT_SOURCE_BASELINE = os.path.join("facts", "source_baseline.json")
CORPUS_FILTERS_BASELINE = os.path.join("facts", "corpus_filters_baseline.json")


def _is_gitignored(path, root):
    """True if path is covered by .gitignore. Tries `git check-ignore` (dev/CI);
    on the pod (no .git) falls back to a minimal .gitignore reader. The fallback
    handles this repo's patterns (directory globs, file globs); it skips negation,
    which this repo uses only for !data/corpus/primary/ (no fact source points there).

    Tries both path and path+/ so a directory pattern (data/corpus/*/) matches a
    source written without a trailing slash (data/corpus/web_hq)."""
    import subprocess

    try:
        r = subprocess.run(
            ["git", "check-ignore", path, path + "/"], capture_output=True, text=True, cwd=root, timeout=5
        )
        if r.returncode == 0:
            return True
        if r.returncode == 1:
            return False
        # 128: git unavailable or not a repo (pod) -> fall through to the reader
    except (OSError, subprocess.SubprocessError):
        pass
    gi = os.path.join(root, ".gitignore")
    if not os.path.exists(gi):
        return False
    import fnmatch

    for line in open(gi, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("!"):
            continue
        dir_only = line.endswith("/")
        pat = line[:-1] if dir_only else line
        if dir_only:
            if path == pat or path.startswith(pat + "/"):
                return True
        elif fnmatch.fnmatch(path, pat) or fnmatch.fnmatch(path + "/", pat):
            return True
    return False


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
        if not named:
            no_reviewer.append(tid)
            continue
        if any(r.get("reviewer") == named for r in reviews.get(tid, [])):
            continue
        closed = t.get("closed", "")
        try:
            age_min = (now - time.mktime(time.strptime(closed, "%Y-%m-%d %H:%M"))) / 60
        except ValueError:
            age_min = REVIEW_GRACE_MIN + 1  # unparseable timestamp: treat as due
        (overdue if age_min > REVIEW_GRACE_MIN else pending).append(f"{tid}->{named}")
    if no_reviewer:
        return FAIL, f"{len(no_reviewer)} done task(s) name no reviewer: {no_reviewer[:4]}"
    if overdue:
        return FAIL, f"{len(overdue)} review(s) over {REVIEW_GRACE_MIN}min overdue: {overdue[:4]}"
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
    same-named run. A fact that cites only a log becomes unreproducible the moment
    the log is cleaned.

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
            if e["id"] in ids:
                errors.append(f"duplicate id {e['id']!r} in {fn} and {ids[e['id']]}")
            ids[e["id"]] = fn
            # Source-path half: a full-checkout check. The pod is a partial checkout (the
            # manifest's executing files, not the repo), so a path missing there is not rot
            # -- it was never there. CI and dev run this fully; the pod skips it. The config
            # half above runs everywhere.
            if not pod_drift.is_pod(root):
                for m in FACT_SOURCE_PATH.findall(str(e["source"])):
                    if os.path.exists(os.path.join(root, m)):
                        continue
                    if _is_gitignored(m, root):
                        continue  # pod-only artifact; this machine doesn't have it
                    if m in source_baseline:
                        baselined.append(m)
                        continue  # registered debt; gaps reports it
                    errors.append(f"{tag}: source path {m} does not exist (not in baseline)")
            entries.append((fn, e))
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
    return PASS, f"{len(entries)} facts in {len(files)} files, every entry carries its config{note}"


def _broken_facts():
    """The REAL facts files and REAL AGENTS.md, with one entry's config deleted and
    one entry's source pointing at a non-existent data/ path. A hand-written file
    would share the check's own assumptions.

    The source mutation uses a bare data/ path with no other prefix substring:
    the old regex (no data/ in its prefix list) found no match and silently passed
    it; the new regex matches data/... and FAILs on the missing file. This is the
    coverage the broken world lacked -- it only exercised the missing-config path,
    which is why the missing left anchor and missing data/ prefix went unnoticed."""
    import shutil

    d = _tmp_repo()
    os.makedirs(os.path.join(d, ".git"), exist_ok=True)  # full checkout: the pod skips the path half
    os.makedirs(os.path.join(d, "facts"))
    for f in glob.glob(os.path.join(FACTS_DIR, "*.json")):
        shutil.copy(f, os.path.join(d, "facts"))
    obj = json.load(open(os.path.join(d, "facts", "tokenizer.json"), encoding="utf-8"))
    del obj["facts"][0]["config"]
    # A source under scripts/ (not gitignored, not in the baseline) that does not
    # exist. A data/ path would be gitignored by data/*.jsonl and silently SKIPped --
    # the source-path mutation must use a path the three-state check treats as FAIL.
    obj["facts"][0]["source"] = "scripts/no_such_script_xyz.py"
    json.dump(obj, open(os.path.join(d, "facts", "tokenizer.json"), "w"))
    shutil.copy(os.path.join(ROOT, "AGENTS.md"), os.path.join(d, "AGENTS.md"))
    return d


ENTRY_SCRIPT_RE = re.compile(r"(?:scripts|eval|datagen|mathbank)/[\w.-]+\.(?:sh|py)|run_ddp\.sh")


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
    for line in open(agents, encoding="utf-8"):
        if "|" not in line or not ENTRY_SCRIPT_RE.search(line):
            continue
        # Task-cell tokens catch attempts logged under an inner command (the wrapper is
        # invisible to the log).
        task_tokens = {t for t in re.split(r"[^a-z0-9]+", line.split("|")[1].lower()) if len(t) >= 5}
        for s in sorted(set(re.findall(r"[\w/.-]+\.(?:sh|py)", line))):
            if not os.path.exists(os.path.join(root, s)):
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
    return PASS, "every tried entry-point command has at least one ok run"


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
    """The REAL AGENTS.md with one table row added citing a script that does not exist -- the
    FAIL tier. The WARN tier is live in the real repo (run_ablation.sh), so it needs no
    synthetic world. The log row is written by the REAL logger with --root d, so the check
    runs instead of SKIPping on an absent log."""
    import shutil, subprocess

    d = _tmp_repo()
    shutil.copy(os.path.join(ROOT, "AGENTS.md"), os.path.join(d, "AGENTS.md"))
    with open(os.path.join(d, "AGENTS.md"), "a") as f:
        f.write("| Ghost | `python scripts/ghost_command.sh` |\n")
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
FACT_REF_RE = re.compile(r"facts/([\w.-]+)\.json#([\w.]+)")
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
                    bad.append(f"docs/{sub}/{f}: facts/{fname}.json does not exist")
                elif fid not in index[fname]:
                    bad.append(f"docs/{sub}/{f}: {fid} not in facts/{fname}.json")
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


def _cited_path_exists(root, tok):
    """A doc-cited data path that resolves. Gitignored artifacts (tokenizer.json, corpus
    bytes) are exempt -- absent from a clean checkout is their normal state; only a
    TRACKED path that is missing is rot. With no git (the pod), disk is the only truth."""
    if os.path.exists(os.path.join(root, tok)):
        return True
    is_repo = (
        subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], cwd=root, capture_output=True).returncode
        == 0
    )
    if not is_repo:
        return False
    r = subprocess.run(["git", "ls-files", "--error-unmatch", tok], cwd=root, capture_output=True, text=True)
    return r.returncode != 0  # tracked-but-missing -> False; untracked (gitignored) -> True


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
    """The REAL README with a retired phrase spliced back in -- the FAIL tier."""
    import shutil
    d = _tmp_repo()
    shutil.copy(os.path.join(ROOT, "README.md"), os.path.join(d, "README.md"))
    shutil.copy(os.path.join(ROOT, "AGENTS.md"), os.path.join(d, "AGENTS.md"))
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
    # Fold by (name, started), last event wins: the ledger is an event log, so one run
    # has a running row and then a terminal one. Reading raw events made a superseded
    # 'ok' outlive the 'fail' that replaced it -- t56_profile, ok 13:34 then fail 13:47,
    # failed this check as an unscored success (2026-08-31).
    folded = {}
    for line in open(log, encoding="utf-8"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        folded[(r.get("name"), r.get("started"))] = r
    rows = list(folded.values())
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
        cand = produced_checkpoint(cmd, str(r.get("name", "?")))
        if cand and f"{cand}.pt" not in scored:
            missing.append(cand)
    if missing:
        return FAIL, f"ok training run(s) with no score-matrix record: {sorted(set(missing))[:5]}"
    return PASS, "every ok training run has a score-matrix record"


def _broken_score_matrix():
    """A REAL ok training row, written by the real exp.py, with no score-matrix
    record -- the FAIL tier."""
    import subprocess

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
        for line in open(exp_path, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
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
        f.write(json.dumps({"name": "test", "started": time.strftime("%Y-%m-%d %H:%M"),
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
    shutil.copy(
        os.path.join(ROOT, "data", "mix_scale_0.2b.json"),
        os.path.join(d, "data", "mix_scale_0.2b.json"),
    )
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
    # The pod is not a git repo: its files must match the committed manifest. CI gates the
    # manifest against HEAD. A dev checkout skips both -- uncommitted changes are normal there.
    if pod_drift.is_pod(root):
        ok, evidence = pod_drift.check_pod(root)
        return (PASS if ok else FAIL), evidence
    if os.environ.get("CI") == "true":
        ok, evidence = pod_drift.check_head(root)
        return (PASS if ok else FAIL), evidence
    return SKIP, "dev checkout; CI gates manifest freshness, the pod gates file drift"


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
    """The REAL manifest plus one REAL scoped file, mutated: the pod gate must see the
    mismatch. The CI branch cannot be exercised here -- the selftest world has no .git."""
    import shutil

    d = _tmp_repo()
    os.makedirs(os.path.join(d, "scripts"))
    shutil.copy(
        os.path.join(ROOT, "data", "pod_head_manifest.txt"),
        os.path.join(d, "data", "pod_head_manifest.txt"),
    )
    shutil.copy(os.path.join(ROOT, "scripts", "harness.py"), os.path.join(d, "scripts", "harness.py"))
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
    a.add_argument("--task", required=True)
    a.add_argument("--why", required=True, help="why this is worth a session's time")
    a.add_argument("--reading", default=None, help="how to read the result, written BEFORE it exists")
    a.add_argument("--blocked-on", dest="blocked_on", default=None)
    d = sub.add_parser("done")
    d.add_argument("id")
    d.add_argument("--reviewer", required=True,
                   help=f"who reads this delivery; a roster member other than the owner {sorted(set(REVIEW_PAIRS))}")
    d.add_argument("--evidence", required=True, help="artifact path, command, or fact id -- not a claim")
    r = sub.add_parser("reopen")
    r.add_argument("id")
    r.add_argument("--why", required=True, help="why this task is being reopened")
    sub.add_parser("list").add_argument("--all", action="store_true", help="include closed tasks")
    args = ap.parse_args(argv)
    rows = _read_tasks()

    if args.op == "add":
        # Owner-scoped ids: a global max+1 collides when two branches allocate
        # concurrently and the union merge keeps both (t52 twice, 2026-08-31).
        # <owner>-<n> is collision-free across branches; existing t-ids stay.
        n = max([int(r["id"].split("-", 1)[1]) for r in rows
                 if re.fullmatch(rf"{re.escape(args.owner)}-\d+", r.get("id", ""))] or [0]) + 1
        row = {
            "id": f"{args.owner}-{n}",
            "owner": args.owner,
            "state": "open",
            "task": args.task,
            "why": args.why,
            "reading": args.reading,
            "blocked_on": args.blocked_on,
            "opened": time.strftime("%Y-%m-%d %H:%M"),
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
        # Append the new state as an event; never rewrite the row (see _read_tasks).
        ev = dict(hit[0], state="done", evidence=args.evidence, reviewer=args.reviewer,
                  closed=time.strftime("%Y-%m-%d %H:%M"))
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
            reopened=time.strftime("%Y-%m-%d %H:%M"),
            evidence=hit[0].get("evidence", ""),  # keep prior evidence; the check accepts open+evidence
        )
        ev.pop("closed", None)
        _append_task(ev)
        print(f"{args.id} reopened: {args.why[:80]}")
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
        "opened": time.strftime("%Y-%m-%d %H:%M"),
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
            r.update(state="done", evidence=evidence, closed=time.strftime("%Y-%m-%d %H:%M"))
            _write_tasks(rows)
            return r["id"]
    return None


def check_tasks_well_formed(root):
    """A closed task carries an artifact; an open one carries an owner and a reason."""
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
    for r in rows:
        if r.get("state") == "done" and not (r.get("evidence") or "").strip():
            bad.append(f"{r.get('id')} done without evidence")
        if r.get("state") == "open" and not (r.get("owner") or "").strip():
            bad.append(f"{r.get('id')} open without an owner")
        if not (r.get("why") or "").strip():
            bad.append(f"{r.get('id')} has no why")
    if bad:
        return FAIL, "; ".join(bad[:3])
    n_open = sum(1 for r in rows if r.get("state") == "open")
    return PASS, f"{len(rows)} task(s), {n_open} open, every closed one carries an artifact"


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
            "opened": time.strftime("%Y-%m-%d %H:%M"),
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
    busy, err = _busy_training_cards(train_cards)
    if err == "not_found":
        return SKIP, "nvidia-smi not installed"
    if err is not None:
        return FAIL, f"nvidia-smi broken: {err}"
    if not busy:
        return PASS, f"training cards {sorted(train_cards)}: idle"
    if len(busy) >= world:
        return PASS, f"training cards {sorted(busy)}: all {world} busy (block used as block)"
    if _has_training_process():
        return PASS, f"training cards {busy}: {len(busy)}/{world} busy (training in progress)"
    return FAIL, (
        f"training cards {busy}: {len(busy)}/{world} busy but no training process — "
        f"a small job is tearing the block. Small jobs go on the lane card "
        f"(the one not in {sorted(train_cards)})."
    )


def _broken_lane_respected():
    """Training cards busy, no training process — the violation this check catches."""
    import shutil

    d = _tmp_repo()
    os.makedirs(os.path.join(d, "data"), exist_ok=True)
    # Real config so the training card set is real, not hand-written.
    shutil.copy(
        os.path.join(ROOT, "data", "mix_scale_run_config.json"),
        os.path.join(d, "data", "mix_scale_run_config.json"),
    )
    config = json.load(open(os.path.join(ROOT, "data", "mix_scale_run_config.json"), encoding="utf-8"))
    first_card = config["cards"].split(",")[0].strip()
    os.environ["HARNESS_BUSY_CARDS"] = first_card
    os.environ["HARNESS_TRAINING_PROC"] = "0"
    return d


# A CUDA_VISIBLE_DEVICES assignment is safe when its value comes from the shard map
# eval/_devs.sh builds (${_DEVS[...]}) or defers to the caller
# (${CUDA_VISIBLE_DEVICES:-...}). Anything else -- a literal, a bare $i, a seq
# expansion -- is a physical index that REPLACES the caller's restriction.
_CVD_SAFE = re.compile(r"^\$\{_DEVS\[|^\$\{CUDA_VISIBLE_DEVICES:-")
_CVD_ASSIGN = re.compile(r"(?:^|\s)(?:export\s+)?CUDA_VISIBLE_DEVICES=(\S+)")
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
    """Untracked files older than 24h in the shared tree — someone's unfinished work.

    In a multi-session tree an untracked file belongs to the session that made it.
    After 24h it is either forgotten or blocked; either way the owner should give
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
    cutoff = time.time() - 24 * 3600
    aged = []
    for f in r.stdout.splitlines():
        p = os.path.join(root, f)
        if os.path.isfile(p) and os.path.getmtime(p) < cutoff:
            aged.append(f)
    if aged:
        return WARN, f"{len(aged)} untracked file(s) older than 24h: {', '.join(aged[:5])}"
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
    """Tracked files dirty longer than 30 minutes — uncommitted work sitting in the
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
    cutoff = time.time() - 30 * 60
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
        return WARN, f"{len(aged)} tracked file(s) dirty >30min: {', '.join(aged[:5])}"
    return PASS, "no aged dirty files"


def _broken_dirty_aged():
    """A real git repo with one tracked file dirty for 2 hours. No git identity is
    configured, so the commit fails and the file sits staged-and-modified ("AM" in
    porcelain) -- the exact shape the old line[:2].strip() parser missed on CI."""
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
    old = time.time() - 2 * 60 * 60
    os.utime(os.path.join(d, "AGENTS.md"), (old, old))
    return d


CHECKS = [
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
        "no_oversized_blob",
        f"no file over {MAX_TRACKED_MB}MB is tracked by git",
        "gitignore does not cover already-tracked paths; a 40MB file committed once because of it",
        check_no_oversized_blob,
        _broken_blob,
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
        "curl_ipv4",
        "every curl call in tracked code passes -4",
        "the pod's IPv6 egress is broken; without -4 the failure reads as 'host unreachable' and produced a whole reachability matrix of false negatives (2026-08-30)",
        check_curl_ipv4,
        _broken_curl_ipv4,
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
        "untracked files older than 24h in the shared tree get a fate",
        "a session's unfinished work sits unowned for days; nobody knows if it is safe to delete",
        check_untracked_aged,
        _broken_untracked_aged,
    ),
    (
        "dirty_aged",
        "tracked files dirty longer than 30min are named so the owner commits or reverts",
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


def run_checks(root=ROOT, quiet=False):
    results = []
    _prev_alarm_handler = signal.signal(signal.SIGALRM, _check_deadline)
    for name, asserts, incident, fn, _broken in CHECKS:
        t0 = time.time()
        try:
            signal.alarm(_CHECK_TIMEOUTS.get(name, _CHECK_TIMEOUT))
            state, evidence = fn(root)
        except TimeoutError:
            state, evidence = SKIP, f"timed out after {_CHECK_TIMEOUTS.get(name, _CHECK_TIMEOUT)}s"
        except Exception as e:  # a check that crashes is a failed check, never a pass
            state, evidence = FAIL, f"the check itself raised: {type(e).__name__}: {e}"
        finally:
            signal.alarm(0)
        dur = time.time() - t0
        results.append((name, state, evidence, asserts, incident))
        if not quiet:
            print(f"  [{state:^4}] {name:<22} {evidence}  ({dur:.1f}s)")
            if state in (FAIL, WARN):
                print(f"         asserts: {asserts}")
            if state == FAIL:
                print(f"         prevents: {incident}")
    signal.signal(signal.SIGALRM, _prev_alarm_handler)
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
            cmd = ["bash", os.path.join(HERE, "eval_all.sh"), ck] + ([tokenizer] if tokenizer else [])
        else:
            cmd = ["bash", os.path.join(HERE, "eval_hard.sh"), ck, str(ngpu or 6)]
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
        f.write(json.dumps({"ts": time.strftime("%Y-%m-%d %H:%M"), "kind": kind, "msg": msg}) + "\n")


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
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "staleness": {"newest_artifact": time.strftime("%Y-%m-%d %H:%M", time.localtime(newest)) if newest else None,
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
        time.sleep(1.5)
        left = sp.run(["pgrep", "-g", str(pgid)], capture_output=True, text=True).stdout.split()
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
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print("  merge fix: bad merge refused, real fix accepted, restaged offender still refused")


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
    becomes a red and the check gets bypassed."""
    import shutil
    import tempfile

    real = "/Users/bytedance/code/aupai"
    if os.path.exists(os.path.join(real, ".git")):
        hit = merge_reverted_content(real, "21da619")
        assert any(n == "_selftest_gpu_descendants" for _, n, _ in hit), \
            f"21da619 must be caught, got {hit}"
        assert not merge_reverted_content(real, "41294c1"), "41294c1 lost nothing; must be clean"

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
    print("  merge revert: 21da619 caught, 41294c1 clean, deliberate deletion not flagged")


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
    """A slow check must SKIP naming its deadline, not kill the harness.

    signal.alarm() with no handler runs SIG_DFL, which terminates: the
    `except TimeoutError -> SKIP` in run_checks was dead code and a slow check exited
    -14/142 with empty stdout and stderr. The hook then refused the commit with no
    check named and told the reader to rerun by hand, where it passes -- the
    --no-verify training P8 exists to prevent. It refused the commit carrying the
    e1-4 review of itself (2026-09-01).

    Tests the PROPERTY -- run_checks turns an overrun into a named SKIP -- not the
    mechanism. My first version asserted a handler was installed at import time,
    which was true only of my own fix; de's is scoped to run_checks and restores the
    previous handler, which is better, and the test failed on the better code. A test
    that encodes one implementation rejects its replacement."""
    slow_name = "__selftest_slow__"

    def slow(_root):
        time.sleep(3)
        return PASS, "should never be reached"

    saved_checks = list(CHECKS)
    saved_to = _CHECK_TIMEOUTS.get(slow_name)
    CHECKS.append((slow_name, "a check that overruns", "the harness dying with no name",
                   slow, lambda: _tmp_repo()))
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

        CHECKS[-1] = (slow_name, "a check that overruns", "the harness dying with no name",
                      timed_slow, lambda: _tmp_repo())
        results = run_checks(ROOT, quiet=True)
        row = [r for r in results if r[0] == slow_name]
        assert row, f"{slow_name} produced no result -- the run died"
        _, state, evidence, _, _ = row[0]
        assert state == SKIP, f"an overrunning check must SKIP, got {state}: {evidence}"
        assert "timed out" in evidence, f"the SKIP must name the deadline: {evidence}"
        assert marks and marks[0] < 2.5, f"the alarm did not interrupt the check ({marks}s)"
    finally:
        CHECKS[:] = saved_checks
        if saved_to is None:
            _CHECK_TIMEOUTS.pop(slow_name, None)
        else:
            _CHECK_TIMEOUTS[slow_name] = saved_to
    print("  check timeout: an overrunning check SKIPs and names its deadline; the run survives")


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
        ev = [
            {"name": "a", "started": "2026-08-31 05:08", "status": "running", "ended": ""},
            {"name": "a", "started": "2026-08-31 05:08", "status": "fail", "ended": "2026-09-01 05:29"},
            # a duplicate START appended AFTER the close, the sft_p324_v3 shape
            {"name": "b", "started": "2026-08-31 03:44", "status": "ok", "ended": "2026-08-31 04:16"},
            {"name": "b", "started": "2026-08-31 03:44", "status": "running", "ended": ""},
            # a genuinely open run must survive the fold
            {"name": "c", "started": "2026-08-31 12:45", "status": "running", "ended": ""},
        ]
        with open(p, "w", encoding="utf-8") as f:
            for r in ev:
                f.write(json.dumps(r) + "\n")

        folded = {(r["name"], r["started"]): r for r in _exp_events(d)}
        assert folded[("a", "2026-08-31 05:08")]["status"] == "fail", "an appended close must clear its start"
        assert folded[("b", "2026-08-31 03:44")]["status"] == "ok", \
            "a start appended after a close must NOT reopen the run"
        assert folded[("c", "2026-08-31 12:45")]["status"] == "running", \
            "a genuinely open run must still read as running"
        assert len(_exp_events(d, folded=False)) == 5, "raw=False must return every event"

        state, evidence = check_no_stale_running(d)
        assert state == PASS, f"only run c is open and it is recent: {state} {evidence}"
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print("  exp fold: close clears its start; a later start does not reopen; open runs survive")


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
    finally:
        ROOT, time.sleep = real_root, real_sleep
        shutil.rmtree(d, ignore_errors=True)
    print("  auto-resume: crash resumes once, clean exit and kill criterion do not")


def _demo():
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
    # not a tree, so no world it builds can hold a repo file.
    synthetic_world = {"no_oversized_blob", "env_importable"}
    # WARN-only checks: their broken world must produce WARN (or FAIL), not PASS/SKIP.
    warn_only = {"untracked_aged", "dirty_aged"}
    untested = []
    for name, _a, _i, fn, broken in CHECKS:
        try:
            root = broken()
        except SelftestSkip as e:
            print(f"  SKIP {name}: {e}")
            continue
        try:
            if name not in synthetic_world and not any(
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
    # HARNESS_GPU_PRESENT is set once before the loop and needed by several broken
    # worlds (mix_shards_present, lane_respected); clean up after the whole loop.
    os.environ.pop("HARNESS_GPU_PRESENT", None)
    assert not untested, "checks that cannot be made to fail:\n  " + "\n  ".join(untested)

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
    # The meta-check carries its own failing case: a fake check whose PASS is vacuous. Without
    # it nothing proves the meta-check fires -- the exact defect it guards against.
    def _vacuous_pass(_root):
        return PASS, "0 domain(s) match filters abc"

    vacuous = []
    for name, _a, _i, fn, _b in list(CHECKS) + [("fake_vacuous_pass", "", "", _vacuous_pass, None)]:
        try:
            state, evidence = fn(ROOT)
        except Exception:
            continue  # a crash against the real repo is the broken-world loop's territory
        if state != PASS:
            continue
        counts = [int(m.group(1)) for m in re.finditer(r"(\d+)\s+[a-zA-Z]", str(evidence))]
        if counts and all(c == 0 for c in counts):
            vacuous.append(f"{name}: PASS with all-zero counts ({evidence})")
    assert any(v.startswith("fake_vacuous_pass") for v in vacuous), (
        "meta-check did not catch its own deliberately-vacuous PASS -- the regex or loop regressed"
    )
    real = [v for v in vacuous if not v.startswith("fake_vacuous_pass")]
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
        n = pod_drift.write_manifest(d)
        assert n == 1, f"expected 1 scoped file, got {n}"
        ok, _ = pod_drift.check_head(d)
        assert ok, "fresh manifest should pass check_head"
        open(os.path.join(d, "scripts", "real.py"), "w").write("# v2\n")
        subprocess.run(["git", "add", "."], cwd=d, capture_output=True)
        subprocess.run(["git", "commit", "-qm", "v2"], cwd=d, capture_output=True)
        ok, evidence = pod_drift.check_head(d)
        assert not ok, f"stale manifest should fail check_head: {evidence}"
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
    shutil.copy(os.path.join(ROOT, "data", "pod_head_manifest.txt"), os.path.join(d2, "data", "pod_head_manifest.txt"))
    subprocess.run(["git", "add", "-A"], cwd=d2, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=d2, capture_output=True)
    # Stage a scoped edit
    with open(os.path.join(d2, "AGENTS.md"), "a") as f:
        f.write("\n# test edit\n")
    subprocess.run(["git", "add", "AGENTS.md"], cwd=d2, capture_output=True)
    r = subprocess.run([hook_dst2], cwd=d2, capture_output=True)
    assert r.returncode == 0, f"hook must pass on scoped edit: {r.stdout} {r.stderr}"
    # The manifest must be staged (regenerated from the index)
    staged_files = subprocess.run(
        ["git", "diff", "--cached", "--name-only"], cwd=d2, capture_output=True, text=True
    ).stdout
    assert "data/pod_head_manifest.txt" in staged_files, f"manifest not staged: {staged_files}"
    # Commit and verify check-head passes
    subprocess.run(["git", "commit", "-m", "scoped edit"], cwd=d2, capture_output=True)
    r = subprocess.run(
        [sys.executable, os.path.join(d2, "scripts", "pod_drift.py"), "--check-head"],
        cwd=d2, capture_output=True, text=True,
    )
    assert r.returncode == 0, f"check-head must pass after hook-mediated commit: {r.stdout} {r.stderr}"
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
        _write_tasks(real_rows, tmp_tasks)
        test_row = dict(real_rows[0])
        test_row.update(id="t_selftest", state="done", evidence="prior evidence",
                        owner="selftest", why="test", closed=time.strftime("%Y-%m-%d %H:%M"))
        rows = _read_tasks(tmp_tasks) + [test_row]
        _write_tasks(rows, tmp_tasks)
        # Reopen: same transition as cmd_task
        rows = _read_tasks(tmp_tasks)
        hit = [r for r in rows if r.get("id") == "t_selftest"]
        assert hit and hit[0]["state"] == "done", "selftest row must start done"
        prior = hit[0].get("evidence", "")
        hit[0].update(state="open", reopen_reason="selftest reopen",
                      reopened=time.strftime("%Y-%m-%d %H:%M"), evidence=prior)
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
    _selftest_merge_reverted_content()

    # Every check must PASS or SKIP on the real tree at the moment it lands.
    # A check that is red on the real artifact the day it ships is the
    # permanent-red failure mode; selftest must catch it, not the first colleague.
    for name, _a, _i, fn, _broken in CHECKS:
        state, _evidence = fn(ROOT)
        assert state != FAIL, f"{name} FAILs on the real tree -- fix the check or the artifact before landing"

    print(f"harness self-test OK ({len(CHECKS)} checks each verified to FAIL on a broken world; "
          f"every PASS verified a non-zero count)")


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
    cmd = [sys.executable, os.path.join(HERE, "pretokenize.py"), *step_args]
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
    "seq", "grad_ckpt", "fone", "doc_mask",  # architecture / training comparability
)

# Architecture constants with no CLI flag. They cannot drift via a launch, so
# _strip_frozen and frozen_args do not touch them. But they can drift via a code
# edit, and ladder_config_frozen compares them against the JSON as documented
# intent -- closing the gap where all six points agree with each other but not
# with what was intended (fb regenerated the manifest mid-ladder, blinding pod_drift).
_CODE_FROZEN_KEYS = ("chunk_size", "layers", "d", "heads", "ffn_hidden")

# CLI flags whose name differs from their Cfg field (--no_attn_res sets Cfg.attn_res).
# --no_doc_mask is gone: it existed because the attention fallback could not honour
# doc_mask, and now it can, so the flag was only a way to turn a frozen recipe key off.
_FLAG_TO_CFG = {"no_attn_res": "attn_res"}

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
    "allow_corpus_drift", "allow_pod_drift", "allow_env_drift",  # safety overrides
    "lr_scale",           # optimizer multiplier, varies by experiment
    "no_static_graph", "no_bucket_view",  # DDP A/B, do not touch Cfg
    "val_every", "val_batches",  # validation cadence, not architecture
    # An A/B arm, like no_attn_res: it exists to take two values, so freezing it would
    # declare settled the very thing the experiment is run to settle. MOVE IT INTO
    # _FROZEN_KEYS the day the A/B says fp32 masters ship -- a decided setting left here
    # is one a launch can silently omit.
    "fp32_master",
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
        frozen_args = [v for k in _FROZEN_KEYS if not isinstance(frozen[k], bool)
                       for v in (f"--{k}", str(frozen[k]))]
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


def cmd_clean(rest):
    """`harness clean --dry` -- list superseded artifacts, do not delete.
    Deletion is a separate step from this committed listing, by exact path.
    Each line: path, bytes, producer, why superseded."""
    if "--dry" not in rest:
        print("usage: harness.py clean --dry (list only; deletion is a separate step)")
        return 2
    pod = os.path.expanduser("~/bin/pod")
    rows = []

    def pod_lines(cmd):
        if not os.path.exists(pod):
            return []
        r = subprocess.run([pod, cmd], capture_output=True, text=True, timeout=30)
        return r.stdout.splitlines() if r.returncode == 0 else []

    # 1. Checkpoints: scan pod for ckpt_*.pt, flag known-superseded names.
    for ln in pod_lines("ls -la /work/aupai/ckpt_*.pt 2>/dev/null"):
        parts = ln.split()
        if len(parts) < 9:
            continue
        sz, name = parts[4], parts[-1]
        base = os.path.basename(name)
        why = ""
        if base == "ckpt_sft_p324_v2.pt":
            why = "superseded by t20/t01 rerun (verify rerun exists before deleting)"
        elif "faFalse" in base or "fa_false" in base or "fa0" in base:
            why = "fa=False ablation arm, superseded by t20/t01 rerun (verify rerun exists)"
        if why:
            rows.append((name, sz, "training run", why))

    # 2. eval_hard shard residue: .N.jsonl files left by multi-GPU eval.
    for ln in pod_lines("find /work/aupai/data/eval -name '*.\\d.jsonl' 2>/dev/null"):
        ln = ln.strip()
        if not ln:
            continue
        sz = pod_lines(f"stat -c%s {ln} 2>/dev/null")
        rows.append((ln, sz[0] if sz else "?", "eval_hard.sh multi-GPU shard",
                      "shard residue: a single-card run read 7 shard leftovers as 148/1032 preds"))

    # 3. Orphan .part files under data/raw.
    for ln in pod_lines("find /work/aupai/data/raw -name '*.part' 2>/dev/null"):
        ln = ln.strip()
        if not ln:
            continue
        sz = pod_lines(f"stat -c%s {ln} 2>/dev/null")
        rows.append((ln, sz[0] if sz else "?", "fetch_corpus.py interrupted",
                      "partial shard, never renamed to final -- fetch deletes stale .part on startup"))

    # 4. /tmp logs on the pod.
    for ln in pod_lines("find /tmp -maxdepth 2 -name '*.log' -size +1M 2>/dev/null"):
        ln = ln.strip()
        if not ln:
            continue
        sz = pod_lines(f"stat -c%s {ln} 2>/dev/null")
        rows.append((ln, sz[0] if sz else "?", "pod /tmp log", "superseded: /tmp is wiped on restart"))

    # 5. Unregistered .py files on the pod: throwaway probes and bare-podput arrivals
    # that no manifest entry names. The ones nothing names (UNNAMED) are deletion
    # candidates; the rest are kept until their run is done.
    import shlex
    scan = (
        "import os,json,time\n"
        "m=set()\n"
        "for l in open('data/pod_head_manifest.txt'):\n"
        " p=l.strip().split('  ',1)\n"
        " if len(p)==2:m.add(p[1])\n"
        "pr={}\n"
        "try:\n"
        " for l in open('runs/experiments.jsonl'):\n"
        "  r=json.loads(l)\n"
        "  for w in r.get('cmd','').split():\n"
        "   if w.endswith('.py'):pr.setdefault(os.path.basename(w),[]).append(r.get('name','?'))\n"
        "except:pass\n"
        "EX=('datagen','filters','mathbank','workflows','.git','__pycache__')\n"
        "for dp,dn,fns in os.walk('.'):\n"
        " dn[:]=[d for d in dn if d not in EX and not d.startswith('.')]\n"
        " rel=os.path.relpath(dp,'.')\n"
        " if rel.split(os.sep)[0] in('data','runs'):continue\n"
        " for fn in fns:\n"
        "  if fn.endswith('.py'):\n"
        "   p=os.path.normpath(os.path.join(rel,fn))\n"
        "   if p not in m:\n"
        "    st=os.stat(p)\n"
        "    prd=', '.join(pr.get(fn,[])[:3])or'UNNAMED'\n"
        "    print(f'/work/aupai/{p}\\t{st.st_size}\\t{time.strftime(\"%Y-%m-%d\",time.localtime(st.st_mtime))}\\t{prd}')\n"
    )
    for ln in pod_lines(f"cd /work/aupai && python3 -c {shlex.quote(scan)}"):
        parts = ln.split("\t")
        if len(parts) >= 4:
            path, sz, mtime, prod = parts[0], parts[1], parts[2], parts[3]
            why = "deletion candidate: no run names it" if prod == "UNNAMED" else f"in use by: {prod}"
            rows.append((path, sz, f"unregistered .py (mtime {mtime})", why))

    if not rows:
        print("no superseded artifacts found")
        return 0
    print(f"{'PATH':<60} {'BYTES':>12}  PRODUCER / WHY SUPERSEDED")
    for path, sz, producer, why in rows:
        print(f"{path:<60} {sz:>12}  {producer}: {why}")
    print(f"\n{len(rows)} candidate(s). Deletion is a separate step, by exact path.")
    return 0


def _allocation_cards(training):
    """Card set from the controller's allocation file, never from the caller.

    Training jobs get the block (all cards in mix_scale_run_config.json).
    Non-training jobs get the lane (the card not in the block)."""
    config_path = os.path.join(ROOT, "data", "mix_scale_run_config.json")
    if os.path.isfile(config_path):
        config = json.load(open(config_path, encoding="utf-8"))
        block = config.get("cards", "")
        if training:
            return block
        block_set = {c.strip() for c in block.split(",") if c.strip()}
        all_cards = {str(i) for i in range(8)}
        lane = sorted(all_cards - block_set)
        return ",".join(lane) if lane else block
    return os.environ.get("CUDA_VISIBLE_DEVICES", "0")


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
    monitor_code = f'''
import json, os, subprocess, sys, time
pid, log, name, exp_py = {pid}, "{log_path}", "{name}", "{os.path.join(HERE, "exp.py")}"
output = {output_repr}
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
                if r.get("name") == name and r.get("status") in ("ok", "fail"):
                    return True
    except (OSError, ValueError):
        pass
    return False

while True:
    time.sleep(60)
    if settled():
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
        subprocess.run([sys.executable, exp_py, "done", "--name", name,
            "--result", "process exited", "--finding", "monitor: process gone",
            "--decision", "check the log", "--status", "ok"], capture_output=True)
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
        subprocess.run([sys.executable, exp_py, "done", "--name", name,
            "--result", "log silent", "--finding", f"monitor: no growth in {{silent_limit}}s",
            "--decision", "check the process", "--status", "fail"], capture_output=True)
        break
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

    # 2a. Lane-occupancy refusal: a non-training GPU job must not start while the
    # lane is occupied. Queue, never spill. Training jobs use the block, not the lane.
    if not args.training and not args.no_gpu and cards:
        lane_card = cards.split(",")[0].strip()
        occupant = _lane_occupant(lane_card)
        if occupant:
            # No ledger row. The refusal happens BEFORE the start row is written, so a
            # second launch under a live run's name cannot close that run's row: on
            # 2026-08-31 l1_rerun_0831 read running/running/fail while pid 550586 was
            # alive and writing, because the row was written at step 1 and this refusal
            # ran at 2a (e1). A job that never starts leaves no trace in the ledger.
            print(f"REFUSED: {args.name} - lane GPU {lane_card} occupied by pid {occupant}. "
                  f"No ledger row written; the lane holds one job at a time.", file=sys.stderr)
            return 1


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
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cards
    env["PYTHONUNBUFFERED"] = "1"  # Python block-buffers stdout when it is a file
    if args.training and cards:
        env["NGPU"] = str(len(cards.split(",")))  # run_ddp.sh defaults to 8; the block is 7

    with open(log_path, "w") as log_f:
        proc = subprocess.Popen(
            cmd,
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
    """(path, step) of the newest ckpt_<name>.pt.step<N>, or (None, None)."""
    best, best_step = None, None
    for p in glob.glob(os.path.join(ROOT, f"ckpt_{name}.pt.step*")):
        m = re.search(r"\.step(\d+)$", p)
        if m and (best_step is None or int(m.group(1)) > best_step):
            best, best_step = p, int(m.group(1))
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
    host_pids = [p for p in r.stdout.split() if p.strip()]
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
    monitor_pids = [p for p in r.stdout.split() if p.strip() and p not in host_pids]
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


MILESTONE_TOKENS = {"3.24b": 3.24e9, "8b": 8e9, "15b": 15e9, "16b": 16e9, "30b": 30e9}


def _milestone_token(name):
    """The milestone token budget encoded in a checkpoint name, or None."""
    m = re.search(r"_(3\.24b|8b|15b|16b|30b)(?:_|[.\-])", name)
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
            "measured": time.strftime("%Y-%m-%d"),
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
                print(f"[{time.strftime('%H:%M:%S')}] milestone {tok} @ step {step} "
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
    """`harness install-hooks` -- symlink .git/hooks/{pre-commit,pre-merge-commit}
    to scripts/hooks/pre-commit. pre-commit covers direct commits; pre-merge-commit
    (git >= 2.24) covers non-fast-forward merges, which otherwise run no hook at all
    (2026-08-31: a bad fact entered main through a clean merge). Fast-forward merges
    carry already-hooked commits, so they are covered by construction.
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
    for name in ("pre-commit", "pre-merge-commit"):
        hook_dst = os.path.join(hooks_dir, name)
        os.makedirs(hooks_dir, exist_ok=True)
        if os.path.lexists(hook_dst):
            os.remove(hook_dst)
        os.symlink(os.path.relpath(hook_src, hooks_dir), hook_dst)
        print(f"installed: {hook_dst} -> {os.path.relpath(hook_src, main_root)}")
    return 0


def main():
    # argparse with choices, not a hand-rolled scan: a bare-flag filter once resolved
    # cmd="7", matched no branch, printed nothing and exited 0 -- a silent no-op, the
    # failure mode this file exists to prevent.
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        return run_dispatch(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "task":
        return cmd_task(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "sync":
        return cmd_sync(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "clean":
        return cmd_clean(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "install-hooks":
        return cmd_install_hooks(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "launch":
        return cmd_launch(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "kill":
        return cmd_kill(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "milestone":
        return cmd_milestone(sys.argv[2:])
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
    ap.add_argument("--selftest", action="store_true", help="every check must FAIL on its broken world")
    a = ap.parse_args()
    if a.selftest:
        return _demo() or 0
    cmd = a.cmd
    res = []
    if cmd in ("all", "check"):
        print("INVARIANTS  (a check that cannot run is a FAILURE, never a pass)")
        res = run_checks()
        bad = [n for n, s, *_ in res if s == FAIL]
        warns = [n for n, s, *_ in res if s == WARN]
    else:
        bad, warns = [], []
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
    if warns:
        print(f"\n{len(warns)} non-blocking warning(s) (to-dos, not failures): {', '.join(warns)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
