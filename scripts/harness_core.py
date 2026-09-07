"""Shared predicates for the harness family: git, ledgers, worktrees, cards, cache paths.

# restartable: pure functions plus two small state files; no long-running work to resume.

WHY THIS MODULE EXISTS, and it is not tidiness. Measured over scripts/harness.py 2026-09-07: 24,703
lines, 469 top-level defs, and 58 of its 183 helpers used by more than one of its four internal
groups (checks / broken worlds / selftests / CLI). Twenty of those are used by three or more. They
are the predicates other tools want too, and having no importable home for them is why the tree
grew three copies of one reduction: board.py:218 and policy_metrics.py:71 each re-implement
_read_tasks's fold and each says so in a comment.

WHAT IS NOT HERE, and why, because the absences are the load-bearing part:

  run_checks                 reads CHECKS, EVIDENCE, TIMEOUT and the timeout maps. A core module
                             that imports the check registry is a cycle; the registry's own runner
                             belongs beside the registry.
  _read_timeout_strikes      read _TIMEOUT_STATE, which is only a path and would move fine. They
  _write_timeout_strikes     stay because harness.py's selftest monkey-patches BOTH through
                             globals()["_read_timeout_strikes"]. Move them and run_checks resolves
                             the real ones here while the patch rewrites harness's copy: the patch
                             goes silently inert and that selftest stops testing anything. A test
                             that cannot fail is worse than a missing test, so the coupling stays.

THIS FILE IS A MOVE, NOT A REWRITE. Every function below was lifted verbatim from harness.py by AST
line span, so the diff is reviewable as a move and the moved code is byte-identical to what was
already reviewed there. harness.py imports these names and re-exports them, so every existing
caller -- including board.py's `from harness import refuse_in_integration_tree` -- keeps working.
"""

import ast
import functools
import json
import os
import re
import subprocess
import sys

# ---- constants the predicates below read, moved with them ----

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
#: How long the reachability probe may take. Short on purpose: this exists to tell an
#: unreachable host from a slow check, and a probe that itself hangs answers neither.
_POD_PROBE_TIMEOUT = 8
#: Cached answer to "can this process reach the pod at all", None until asked.
_POD_REACHABLE = None
_SKIP_DIRS = {".git", "data", "runs", "node_modules", "__pycache__", ".venv", "venv"}
#: A trailing `.` is sentence punctuation, not part of the id. Fact ids contain dots
#: (`dq.audit.protocol_400`), so the id class has to accept them -- but `[\w.]+` is greedy
#: and swallows the period that ends the sentence, turning a live citation into
#: `be.math_v2_likelihood_twin.` and a FAIL that names an id nobody wrote. Latent when
#: found, not live: zero of the 72 current doc citations end a sentence, so the check would
#: have gone wrong on the first one that did (de-16, 2026-09-02, after writing the same bug
#: into a one-off scanner and getting a false positive out of it -- the repo's own
#: greedy-regex-over-JSONL lesson, one field over).
FACT_REF_RE = re.compile(r"facts/([\w.-]+)\.json#([\w.]*[\w])")
TASKS_PATH = os.path.join(ROOT, "runs", "tasks.jsonl")
#: How many following lines a sentence may continue onto. 2, not unbounded: the sha has to be
#: near the number it anchors, and an unbounded reach becomes the whole-file search that would
#: accept a sha from an unrelated paragraph.
_CITE_WRAP_LINES = 2
# Anchored on an arm SUFFIX or an explicit mem_ marker, never on a leading m1: see the note
# in check_memory_diag_fresh. Module-level so the check and _arm_id share one definition.
_ARM_RE = re.compile(r"(^|_)mem_m[123]([_-]|$)|(^|_)m[123]([_-]|$)", re.I)
# A cards[] value that marks the card as belonging to the RL team rather than to us. The map's
# keys are specs ("1-4", "5") and its values are prose, so this is the only machine-readable
# signal in it. Matched case-insensitively on the phrase, not on an exact string, because the
# entries are written by hand ("RL TEAM (tileRL) -- not ours").
_RL_TEAM_RE = re.compile(r"\bRL[ _-]?TEAM\b", re.I)


def pod_reachable():
    """Whether ~/bin/pod answers at all, measured ONCE per process. (bool, why).

    6e's ruling, 2026-09-04, from a commit refused twice while the tn tunnel flapped: an auth=pod
    check with the pod unreachable must SKIP naming the reason, never time out and block a commit it
    has nothing to say about. Three states present identically as a TimeoutError today and the
    deadline mechanism cannot separate them:

      a hang            what the deadline exists for. Raise nothing, fix the check.
      cost growth       snapshot_logs_say_so_at_the_tail reads 84 tracked logs at ~0.11s each and
                        will cross ANY fixed deadline as runs/ fills. A bigger number buys time,
                        not a fix.
      unreachable host  nothing to measure. Not the check's fault and not the repo's.

    Raising deadlines is the wrong lever for two of the three, which is why this exists rather than
    another _CHECK_TIMEOUTS entry. On an unreachable pod the check SKIPs and its strike counter is
    NOT incremented -- a check that never got to run has not struck out, and banking strikes against
    a dropped tunnel is what produced "has not actually run since" on two checks that both pass by
    hand (2.9s and 9.3s, measured at load avg 6.5).

    ONE PROBE PER PROCESS, cached. `harness check` runs 83 checks and ~20 are auth=pod; probing per
    check would multiply a dead tunnel's timeout by twenty. The cache is per-process on purpose: a
    tunnel that comes back mid-run is not worth the complexity, and the next run re-probes.

    Cheap by construction -- `true` in the container, not a file read -- so it measures the tunnel
    and nothing else.
    """
    global _POD_REACHABLE
    if _POD_REACHABLE is not None:
        return _POD_REACHABLE
    # THE SEAM IS A PATH, NOT A REWRITTEN ~/bin/pod. 6e specified the world as "point
    # ~/bin/pod at a dead host", which means editing the operator's real wrapper -- the one
    # every session and five tracked scripts depend on -- so a selftest that crashed midway
    # would leave the machine unable to reach the pod. HARNESS_POD_BIN names a DIFFERENT
    # executable instead, and the world builds a real one: a two-line script that sleeps past
    # the probe deadline, so the TimeoutExpired branch is exercised by an actual timeout rather
    # than by a stubbed exception (de-25, 2026-09-05).
    # THE SEAM IS DELIBERATELY ONLY THE PROBE. Twelve other sites in this file resolve
    # ~/bin/pod directly and stay that way: HARNESS_POD_BIN exists so a selftest can make the
    # REACHABILITY question answer no, and a fixture that could redirect every pod read would
    # be able to fake the pod's content, which is the thing those checks exist to read.
    pod = os.environ.get("HARNESS_POD_BIN") or os.path.expanduser("~/bin/pod")
    if not os.path.exists(pod):
        _POD_REACHABLE = (False, "~/bin/pod is not installed on this machine")
        return _POD_REACHABLE
    try:
        r = subprocess.run([pod, "true"], capture_output=True, text=True,
                           timeout=_POD_PROBE_TIMEOUT)
        ok = r.returncode == 0
        why = "" if ok else (f"~/bin/pod exited {r.returncode}: "
                             f"{(r.stderr or r.stdout or '').strip()[:80]}")
    except subprocess.TimeoutExpired:
        ok, why = False, f"~/bin/pod did not answer within {_POD_PROBE_TIMEOUT}s (tunnel down)"
    except OSError as e:
        ok, why = False, f"~/bin/pod could not be run: {e}"
    _POD_REACHABLE = (ok, why)
    return _POD_REACHABLE

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
            # And the monitor rule, duplicated for the same reason as the retraction rule above: a
            # fallback that folds differently from the real fold is the divergence this function
            # exists to end. A monitor's close reports process state, not a result, so a human's
            # close outvotes it regardless of union-merge order (4c, 2026-09-07).
            if (prev is not None and r.get("writer") == "monitor"
                    and prev.get("status") in ("ok", "fail") and prev.get("writer") != "monitor"):
                continue
            out[key] = r
        return list(out.values())

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
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=d, capture_output=True)
    for sub in os.listdir(os.path.join(ROOT, "data")):
        src, dst = os.path.join(ROOT, "data", sub), os.path.join(d, "data", sub)
        if not os.path.exists(dst):
            os.symlink(src, dst)
    for f in os.listdir(os.path.join(ROOT, "runs")):
        src, dst = os.path.join(ROOT, "runs", f), os.path.join(d, "runs", f)
        if not os.path.exists(dst):
            os.symlink(src, dst)
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

def refuse_in_integration_tree(what, path=None):
    """A ledger writer refuses in the integration tree.

    AGENTS.md's rule -- run `harness task` and `harness friction` in your worktree, never in
    the integration tree -- was prose, and prose is what people break for cause. Two rows
    landed in the integration tree ten minutes apart on 2026-09-05: b0's task row, then 44's
    board row. Nobody was careless; the tree is where you end up when a command refuses to run
    in a worktree, and the rule's coverage row said so ("the invoking directory is a shell fact
    no artifact records").

    THE PREDICATE IS "THIS IS THE MAIN WORKTREE OF A COMMON GIT DIR THAT HAS LINKED WORKTREES",
    in scripts/integration_tree.py, shared with scripts/hooks/pre-commit. It was `branch ==
    "main"` for the first hours of its life and that was wrong within the day: the integration
    tree was DETACHED on purpose (tilerl's flip, main 0425accb, 2026-09-05), which makes
    symbolic-ref answer "HEAD" and turned all three guards OFF in the one tree they exist for.
    A branch is a label anyone can change; being the tree other worktrees hang off is
    structural. Read integration_tree.py before touching the predicate -- each of its clauses
    is there because dropping it was measured to break a real world, and the second clause in
    particular keeps CI and all 27 of this file's mkdtemp fixtures out.

    THE CONSEQUENCE THIS PREVENTS IS NOT THE ROW, IT IS THE DIRTY LEDGER. The row itself is
    valid content; what breaks is that the integration tree's pre-commit hook refuses the
    commit, so the append sits uncommitted in the tree every other session merges through, and
    the next merge aborts on it. That is why the refusal is at the write and not at the commit:
    by the time the hook speaks, the file is already dirty.

    NO AUPAI_CONTROLLER LIFT. It existed while merge_main.sh:425 appended a friction row from
    the integration tree; the new merge_main writes its rows from the caller's worktree, so
    nothing legitimately appends there any more (4c, 2026-09-05, confirmed as intent -- their
    own rulings come from ../aupai-fb, which is a linked worktree and reads False).

    Fails OPEN where git cannot answer -- no repository, git absent, git erroring. Such a tree
    is not the integration tree by any definition, and refusing there would break every
    temp-dir fixture and every CI runner. A guard whose broken state blocks the write is a
    guard people disable. An UNIMPORTABLE predicate is different and says so loudly rather
    than silently: with the tree detached there is no branch test left to fall back to, so a
    swallowed ImportError would leave the integration tree wholly unguarded (tilerl's reasoning
    in the hook, adopted here).
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(path))) if path else ROOT
    try:
        from integration_tree import is_integration_tree
    except Exception as e:
        print(f"NOTE: the integration-tree guard could not load "
              f"({type(e).__name__}: {e}); writing {what} unguarded -- check by hand that this "
              f"is not the integration tree ({root})", file=sys.stderr)
        return False
    if not is_integration_tree(root):
        return False
    print(f"refusing: {what} would write the ledger of the INTEGRATION TREE ({root}).\n"
          f"  That tree's pre-commit hook refuses the commit, so the row would sit dirty in the\n"
          f"  tree every session merges through, and abort the next merge.\n"
          f"  Run this in your own worktree; the ledgers merge by union, so nothing is lost.",
          file=sys.stderr)
    return True

def fold_by_id(rows, key="id"):
    """Last event per key wins, in first-insertion order. THE fold for every runs/*.jsonl ledger.

    ONE IMPLEMENTATION, and it exists because there were three. `_read_tasks` had it inline,
    scripts/board.py re-derived it with the comment "the same fold exp.py and _read_tasks use", and
    scripts/policy_metrics.py re-derived it with "Last row per id wins, like harness._read_tasks".
    Two copies documented as copies, agreeing with the original by coincidence rather than by
    construction. There was no importable home for it, so this module is what makes deleting them
    possible (de-71, 2026-09-07).

    A ROW WITH NO KEY IS STILL FOLDED, under the key `None`, because that is what _read_tasks
    already did and changing it here would change its answer silently. policy_metrics's copy
    DROPPED such a row; that divergence is recorded at its import site. A ledger with keyless rows
    is ledgers_one_line_per_row's problem, not this function's.
    """
    folded = {}
    for r in rows:
        folded[r.get(key)] = r  # dict preserves first-insertion order; the value is the last event
    return list(folded.values())


def _read_tasks(path=None, raw=False, index_root=None):
    """The register, folded by id: last row for an id wins.

    The file is an EVENT LOG, not a table. `task done`/`reopen` append a new row
    carrying the same id and the new state instead of rewriting the old one,
    because runs/*.jsonl merges by union: when two branches rewrite the same row,
    union keeps BOTH and the register grows a duplicate id (2026-08-31, t39 and
    t40 -- an open row and a done row for each, and tasks_well_formed failed the
    merge). Appends from different branches union cleanly and fold to the same
    state whichever order they land in.

    raw=True returns every event, for the checks that must see collisions.

    index_root reads the STAGED register (`git show :runs/tasks.jsonl`) instead of the
    working tree's. For a check that also resolves fact citations against the index, the
    two sides must come from ONE tree: reading rows from the working tree and facts from
    the index means the check judges a tree that exists nowhere (b0, 2026-09-06). Falls
    back to the file when the path is not in the index -- a fresh clone, a check run
    outside a repo, or simply an unstaged register, none of which are errors here.
    """
    p = path or TASKS_PATH
    text = None
    if index_root is not None:
        rel = os.path.relpath(p, index_root)
        r = subprocess.run(["git", "-C", index_root, "show", f":{rel}"],
                           capture_output=True, text=True)
        if r.returncode == 0:
            text = r.stdout
    if text is None:
        if not os.path.exists(p):
            return []
        text = open(p, encoding="utf-8").read()
    # A concurrent append can be observed mid-write: the reader sees a torn line and
    # json.loads raises, so `harness check` failed inside a hook and passed 20 s later
    # by hand -- a flake that reads as a real refusal (fb, 2026-08-31). Skip a line
    # that will not parse; ledgers_one_line_per_row is what judges malformed rows, and
    # it runs when nobody is mid-write.
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if raw:
        return rows
    return fold_by_id(rows)

def _write_tasks(rows, path=None):
    """Rewrite the register in place. Guarded like _append_task, and for a stronger reason:
    a rewrite in the integration tree dirties the WHOLE file rather than one line, so the
    merge it aborts cannot be resolved by dropping a row."""
    p = path or TASKS_PATH
    if refuse_in_integration_tree(f"rewriting {os.path.basename(p)}", path=p):
        raise SystemExit(1)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# MEMOISED PER ROOT, and these two dicts MUST be the objects harness.py mutates, not copies of them.
# harness.py's own selftest calls `_MAIN_TOUCHED.pop(d, None)` between its two worlds to force a
# re-read; a `from harness_core import _MAIN_TOUCHED` binds the same dict, so the pop is seen here.
# Rebuilding them in harness.py instead would give the selftest a dict nothing reads and its
# world-2-then-world-1 ordering would silently test one world twice (de-71, 2026-09-07).
_MAIN_WHEN = {}


_MAIN_TOUCHED = {}


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
        # main_ref, not a bare "main": on a CI pull_request build the bare form exits 128 and this
        # returns an EMPTY map, which reads as "no commit on main has a date" -- silent, unlike
        # _main_touched's raise, so every time-comparison in _commit_delivers quietly stopped
        # applying rather than failing loudly.
        r = subprocess.run(["git", "-C", root, "log", main_ref(root) or "main",
                            "--format=%H %cd", "--date=format-local:%Y-%m-%d %H:%M"],
                           capture_output=True, text=True, env={**os.environ, "TZ": "UTC"})
        _MAIN_WHEN[root] = dict(ln.split(" ", 1) for ln in r.stdout.splitlines() if " " in ln)
    return _MAIN_WHEN[root]

_MAIN_REF = {}


def main_ref(root):
    """The ref that IS main in `root`: "main", "origin/main", or "refs/remotes/origin/main".

    None when no ref named main resolves at all -- an unreadable tree, which callers must
    treat as such rather than as an empty history.

    WHY THIS IS SHARED. actions/checkout on a pull_request build fetches the base as
    refs/remotes/origin/main and creates NO local main, so a bare `git log main` /
    `git rev-parse main` exits 128 on a repository that reads perfectly. Three call sites had
    the bare form and each fails differently: _main_touched raised, _main_when returned an
    empty map, and _broken_tasks_closed_by_commit BUILT A WORLD WITH NO MAIN -- which is the
    one that cost the second CI round, because fixing only the reader left the fixture handing
    it a mainless repo and the raise fired from the world instead of the code under test.

    A LOCAL main WINS wherever it exists, so a laptop, the pod, and a normal push build are
    unchanged; the remote-tracking forms are the CI fallback only.
    """
    if root not in _MAIN_REF:
        found = None
        for ref in ("main", "origin/main", "refs/remotes/origin/main"):
            r = subprocess.run(["git", "-C", root, "rev-parse", "--verify", "--quiet", ref],
                               capture_output=True, text=True)
            if r.returncode == 0 and r.stdout.strip():
                found = ref
                break
        _MAIN_REF[root] = found
    return _MAIN_REF[root]


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
        # WHICH REF IS `main` HERE: see main_ref. A local main wins; on a CI pull_request build
        # only refs/remotes/origin/main exists, and the bare form exited 128 there -- reported
        # faithfully by the raise below as an unreadable main, which failed every PR check job
        # on a tree that reads fine (4c 2026-09-07: 3b's PR #1, jobs 34088976228 / 34089010051,
        # blocking every session's PRs on the day the flip landed).
        ref = main_ref(root) or "main"
        r = subprocess.run(["git", "-C", root, "log", ref, "-m",
                            "--name-only", "--format=%x00%H"],
                           capture_output=True, text=True)
        # A NONZERO rc RAISES. It used to be discarded: a tree with no readable `main` exits 128
        # with an empty stdout, the parse below yields {}, and the map says "main touches
        # nothing" rather than "main could not be read". Both consumers then report every closed
        # task as undelivered -- measured 2026-09-06 on a clone whose local main was deleted,
        # `FAIL 156 of 156 ... does not reach main`, whose text sends the reader to the worktree
        # while the cause is the missing ref.
        #
        # THE :7750 GUARD CANNOT COVER THIS, which is why the fix belongs here and not there.
        # That guard exists for this function going blind, but keys on "the sha is in the map
        # with no paths": `blind = [s for s in merges if s in touched and not touched[s]]`. On an
        # empty map no sha is in it, so blind == [] and `0 > len(merges)//2` is false. Deleting
        # its `if touched:` would not help -- partial blindness and total blindness look different
        # in the same structure, and it only ever described the first.
        if r.returncode != 0:
            raise RuntimeError(
                f"git log main failed in {root} (exit {r.returncode}): no ref named main, "
                f"origin/main or refs/remotes/origin/main resolves: {r.stderr.strip()[:200]}")
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
        # THE INDEX, NOT HEAD. This check runs from the pre-commit hook, where the content the
        # commit will carry is staged and HEAD is the parent. Reading HEAD made a MERGE that
        # brings the cited fact in fail at the exact moment it resolves the citation: de,
        # 2026-09-06, e1-44 citing eff.moe48_dense_step_cost_ratio -- absent at HEAD, present
        # at MERGE_HEAD and in the index, so `git merge main` could not be committed by anyone
        # until the fact reached HEAD, which only that commit could do. `git show :<path>` is
        # the staged blob and equals HEAD's for an unmodified file, so CI reads the same thing.
        r = subprocess.run(g + ["show", f":facts/{fname}.json"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            return f"evidence cites facts/{fname}.json#{fid} but that file is not in the index"
        try:
            ids = {e.get("id") for e in json.loads(r.stdout).get("facts", [])}
        except ValueError:
            return (f"evidence cites facts/{fname}.json#{fid} but that file is not valid JSON "
                    f"in the index")
        if fid not in ids:
            return (f"evidence cites facts/{fname}.json#{fid} but that id is not in the file "
                    f"being committed -- the citation does not resolve")
    when = _main_when(root).get(full, "")
    if closed and closed >= "2026-09-02" and when and when > closed[:16] + ":59":
        return (f"{sha[:8]} was committed at {when}, after the row closed at {closed} -- "
                "a delivery cited after the fact is a repair of the register, not the delivery; "
                "reopen and close again on the commit that exists")
    return ""

def _cite_sentence(line, pos, following=()):
    """The sentence a citation at `pos` sits in, for the sha search.

    Sentence, not line: the ruling says "the same sentence names a sha", and a comment
    wraps across lines, so a line-scoped search would miss a sha one line below the number
    and a whole-file search would accept a sha from an unrelated paragraph. Bounded by
    sentence punctuation, falling back to the line when there is none.

    A LINE BREAK INSIDE A BLOCK CONTINUES THE SENTENCE (4c's second finding, 2026-09-07; the
    rule is stated here because either answer had to be). Measured: `# the cast at
    train.py:2315 AT` / `# 169da865 held the fp8 branch.` -- the sha is on the second line, so
    the line-scoped search found none and the check reported a correctly ANCHORED citation as
    bare debt. That is the same class of false red as the position-keyed baseline: a spurious
    FAIL on a citation nobody touched, and here it also pushes an author toward deleting the
    one spelling that cannot rot.

    The continuation is bounded three ways, so it cannot become the whole-file search:
      - it stops at the first sentence punctuation, as on the citing line;
      - it stops after _CITE_WRAP_LINES following lines;
      - it stops at a line of a different KIND -- a comment continues onto a comment, prose
        onto prose, and neither onto a blank line. A `#` block ending and code resuming is a
        different kind, so the sha in the next statement is not read as part of the sentence.
    """
    left = max((line.rfind(c, 0, pos) for c in (". ", "; ", "! ")), default=-1)
    ends = [r for r in (line.find(c, pos) for c in (". ", "; ", "! ")) if r != -1]
    if ends:
        return line[left + 1:min(ends)]
    sent = line[left + 1:]
    is_comment = line.lstrip().startswith("#")
    for nxt in list(following)[:_CITE_WRAP_LINES]:
        stripped = nxt.strip()
        if not stripped or stripped.startswith("#") != is_comment:
            break
        sent += " " + stripped.lstrip("#").strip()
        if any(c in stripped for c in (". ", "; ", "! ")) or stripped.endswith("."):
            break
    return sent

def _arm_id(name):
    """'m1' from b0_mem_m1, m1_probe, mem_m1_resume; None from anything that is not an arm.

    Delegates to memory_diag._arm_key, which the launch monitor also calls. Two copies of
    this predicate is how the run side and the monitor side end up disagreeing about which
    names are arms, and a disagreement here is silent: both sides return a plausible answer.
    Falls back to the local pattern only if memory_diag is unimportable, which happens in a
    partial selftest world.
    """
    try:
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        import memory_diag as _md
        return _md._arm_key(name)
    except Exception:
        m = _ARM_RE.search(str(name or ""))
        if not m:
            return None
        return re.search(r"m[123]", m.group(0), re.I).group(0).lower()

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

def _card_map(root=None):
    """{card index: the grant file's prose for it}, expanding the map's range keys.

    cards[] is keyed by SPEC, not by index: today's file has "1-4" and "5" as separate keys, so
    a reader indexing it by str(card) finds nothing for card 2 and would conclude the card is
    unassigned. Returns {} when the file is absent or unreadable -- callers must treat an empty
    map as "no information", never as "no card is claimed by anyone".
    """
    root = ROOT if root is None else root
    try:
        with open(os.path.join(root, "runs", "card_assignment.json"), encoding="utf-8") as fh:
            a = json.load(fh)
    except (OSError, ValueError):
        return {}
    out = {}
    for spec, note in (a.get("cards") or {}).items():
        for c in _expand_cards(spec):
            out[c] = note
    return out

def _aupai_cards(root=None):
    """(ours, theirs, unlisted_is_unknown) from the grant file's cards[] map.

    ours = every listed card whose note does NOT mark it RL TEAM. theirs = the RL-team ones.
    A card absent from the map is in NEITHER set: the map is the only statement of ownership
    there is, and "not mentioned" is not a grant (idle is not a grant, and neither is silence).
    """
    m = _card_map(root)
    ours = sorted(c for c, note in m.items() if not _RL_TEAM_RE.search(str(note)))
    theirs = sorted(c for c, note in m.items() if _RL_TEAM_RE.search(str(note)))
    return ours, theirs, m

def _close_row(name, status, result, finding, decision, root=None, writer=""):
    """Close an exp row. `root` exists for the selftest: exp.py takes no ambient
    override (the ledger gets no env var), so a test that cannot redirect it writes
    into the real ledger -- which is exactly what happened (four 'arts' rows,
    2026-08-31, one pair sharing an identity that then failed the sync guard).

    `writer` marks who closed it. Every call from the auto-resume supervisor passes
    `monitor`, because those rows report PROCESS STATE (the pid returned 0, the pid
    vanished) rather than what the run measured. exp.fold then lets a human's close
    outvote them regardless of union-merge order -- without it, `exit 0 / monitor:
    process exited cleanly` can silently replace `val 2.884, stop rule 4 tripped`
    when two branches' rows are unioned (4c, 2026-09-07). Default empty: a row with
    no writer is a human's, which is what every row already in the ledger is.

    `--root` GOES BEFORE THE SUBCOMMAND, and until 2026-09-07 this function appended it after.
    exp.py declares it on the top-level parser, so `exp.py done ... --root X` exits 2 with
    `unrecognized arguments` -- and with capture_output=True that went nowhere. Every
    _close_row(root=...) call wrote NOTHING and returned as if it had worked. No caller in the
    tree passed root, so no production close was affected; it was found by the first test that
    tried to use the parameter for the purpose the docstring gives it. The failure is now loud
    rather than swallowed, for the reason the docstring already implies: a close that silently
    does not happen is the defect its callers exist to prevent.
    """
    cmd = [sys.executable, os.path.join(HERE, "exp.py")]
    if root:
        cmd += ["--root", root]
    cmd += ["done", "--name", name,
            "--result", result, "--finding", finding, "--decision", decision, "--status", status]
    if writer:
        cmd += ["--writer", writer]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"WARN: could not close the row for {name}: exp.py exited {r.returncode}: "
              f"{(r.stderr or r.stdout).strip()[-300:]}", file=sys.stderr)
    return r.returncode == 0
