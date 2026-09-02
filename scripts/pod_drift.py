#!/usr/bin/env python
"""The pod is not a git repo and code arrives by hand-push, so drift between HEAD and
the pod is invisible without a reference. data/pod_head_manifest.txt is that reference:
sha256 of every file that executes on the pod, committed with the code.

Two gates read it:
  - the pod (no .git): every file the manifest names must exist and match. harness
    check runs this, and train.py's startup raises on it, so drift shouts before any
    training instead of waiting for someone to remember pod_sync_check.sh.
  - CI (has .git, CI=1): the committed manifest must match HEAD, so a scoped change
    cannot land without regenerating it.

The pod gate also scans in reverse: a .py file under the repo root that no manifest
entry names is reported (scripts/_audit_anchor.py arrived by bare podput and ran
unregistered for a day before anything noticed). Reported, not failed -- the pod
legitimately holds throwaway probes -- but it must appear in the output.

A dev checkout (has .git, no CI) skips both: uncommitted changes are normal there.

Usage:
  python scripts/pod_drift.py --write        # regenerate the manifest from HEAD
  python scripts/pod_drift.py --check        # pod gate: files here vs the manifest
  python scripts/pod_drift.py --check-head   # CI gate: manifest vs HEAD
  python scripts/pod_drift.py --list-scoped  # the file set, one per line
  python scripts/pod_drift.py --selftest     # reverse scan on a real-shaped world
"""
import hashlib
import os
import re
import subprocess
import sys

# POD_DRIFT_ROOT: the hook runs a STAGED COPY of this file from the gitdir
# (<common>.git/worktrees/<name>/pod_drift_staged.py in a linked worktree), where
# __file__-relative ROOT resolves to <common>.git/worktrees -- not the worktree.
# The manifest then writes to a phantom path and the hook's check reads the
# unregenerated worktree manifest, refusing a commit that deletes scoped files
# (measured 2026-09-02, 44-15). The hook passes the real root explicitly.
ROOT = os.environ.get("POD_DRIFT_ROOT") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "data", "pod_head_manifest.txt")

# Files that execute on the pod (pretrain -> score flow), plus the files that DECIDE
# what the pod runs on: the hooks and the corpus filters. "Never runs there" was the
# wrong test -- filters/ is read to build the data, and a file nobody executes can still
# change the result.
SCOPE = [
    "*.py",
    "*.sh",
    "scripts/hooks/*",
    "data/mix_*.json",
    "data/tokenizer.json",
    "facts/*.json",
    "scripts/*.json",
    "runs/*.jsonl",
    "AGENTS.md",
    "docs/standards/*.md",
    ":!scripts/pod_sync_check.sh",
]
# workflows/ only: it holds no code the pod runs. filters/ and mathbank/ were excluded
# with it and should not have been -- datagen reads filters/ to BUILD the corpus, so a
# filters/ file that differs silently changes the data, and fp_filters hashes the whole
# directory. Measured 2026-09-01: pod's filters/pass3_garbage.py was 5 CCI3-HQ rules
# behind main and the corpus was built with it (fb: ~0.25% residue, not rebuilt), while
# pod_drift read zero drift throughout.
EXCLUDE_DIRS = ("workflows",)

# git INHERITS these from the caller. The hook runs the selftests below with GIT_DIR set
# to the committing worktree's gitdir and GIT_INDEX_FILE to its temp index, so a `git
# init` in a throwaway directory reconfigures the REAL repo instead: with GIT_DIR
# pointing at a worktree gitdir it sets core.bare=true on the shared repo, which breaks
# every git command for every session until someone unsets it. A selftest must build its
# world from nothing the caller is holding.
_GIT_ENV_VARS = ("GIT_INDEX_FILE", "GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR",
                 "GIT_OBJECT_DIRECTORY", "GIT_CONFIG", "GIT_CONFIG_GLOBAL")


def _clean_git_env():
    return {k: v for k, v in os.environ.items() if k not in _GIT_ENV_VARS}

# Job-class entry points: BFS from these through imports derives the class of
# every manifest file. Priority: training > eval > corpus > docs.
_ENTRY_POINTS = {
    "training": ["train.py", "sft.py", "sft_math.py", "run_ddp.sh", "scripts/run_sft.sh"],
    "eval": ["eval/run_eval.py", "eval/math_hard.py", "eval/eval_hard.sh",
             "eval/eval_all.sh", "eval/score_matrix.py"],
    "corpus": ["datagen/fetch_corpus.py", "datagen/clean_corpus.py",
               "datagen/count_cleaned_code.py"],
}


def _imports(path):
    """Files this file imports or cites. Python files follow Python imports only;
    shell files follow script citations. Mixing the two makes docstring mentions
    in a Python file look like runtime dependencies."""
    try:
        text = open(path, encoding="utf-8", errors="ignore").read()
    except OSError:
        return set()
    deps = set()
    if path.endswith(".py"):
        for m in re.finditer(r"^\s*(?:from|import)\s+([\w.]+)", text, re.MULTILINE):
            mod = m.group(1)
            for cand in (mod.replace(".", "/") + ".py",
                         os.path.join("scripts", mod.split(".")[-1] + ".py"),
                         os.path.join("eval", mod.split(".")[-1] + ".py"),
                         os.path.join("algorithms", mod.split(".")[-1] + ".py"),
                         os.path.join("datagen", mod.split(".")[-1] + ".py"),
                         mod.split(".")[-1] + ".py"):
                if os.path.isfile(os.path.join(ROOT, cand)):
                    deps.add(cand)
                    break
    elif path.endswith(".sh"):
        for m in re.finditer(r"(?:scripts|eval|algorithms)/[\w./-]+\.(?:py|sh)", text):
            if os.path.isfile(os.path.join(ROOT, m.group(0))):
                deps.add(m.group(0))
    return deps


def _classify_files():
    """Job class for every file in the manifest, derived from BFS over import edges
    from the class entry points. Files not reached by any BFS are 'docs'."""
    classes = {}  # path -> class (first assignment wins by priority)
    for job_class in ("training", "eval", "corpus"):
        seen = set()
        queue = [e for e in _ENTRY_POINTS[job_class] if os.path.isfile(os.path.join(ROOT, e))]
        while queue:
            f = queue.pop()
            if f in seen or f in classes:
                continue
            seen.add(f)
            classes[f] = job_class
            for dep in _imports(os.path.join(ROOT, f)):
                if dep not in seen and dep not in classes:
                    queue.append(dep)
    return classes


def scoped_paths(root=ROOT):
    out = subprocess.run(
        ["git", "ls-files", *SCOPE], cwd=root, capture_output=True, text=True, check=True
    ).stdout.split()
    return sorted(p for p in out if not p.startswith(EXCLUDE_DIRS))


def _sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha_head(root, path):
    r = subprocess.run(
        ["git", "show", f"HEAD:{path}"], cwd=root, capture_output=True
    )
    if r.returncode != 0:
        return None
    return _sha_bytes(r.stdout)


def sha_index(root, path):
    """sha256 of the staged blob (git show :path). Returns None if not in the index."""
    r = subprocess.run(
        ["git", "show", f":{path}"], cwd=root, capture_output=True
    )
    if r.returncode != 0:
        return None
    return _sha_bytes(r.stdout)


def sha_disk(path):
    with open(path, "rb") as f:
        return _sha_bytes(f.read())


def write_manifest(root=ROOT):
    classes = _classify_files()
    lines = [f"{sha_head(root, p)}  {p}  {classes.get(p, 'docs')}" for p in scoped_paths(root)]
    with open(MANIFEST, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return len(lines)


def write_manifest_index(root=ROOT):
    """Manifest from the index (staged blobs), not HEAD. Used by the pre-commit
    hook so the committed manifest matches HEAD after the commit lands.

    Entries unchanged since HEAD are reused from HEAD's manifest (sha only; the
    class is recomputed -- a changed import elsewhere can reclassify an unchanged
    file). Rehashing all 174 paths on every commit made the hook take >2min and
    --no-verify became a habit (4a7dd56, 2026-08-31)."""
    classes = _classify_files()
    head = {}
    r = subprocess.run(
        ["git", "show", "HEAD:data/pod_head_manifest.txt"], cwd=root,
        capture_output=True, text=True,
    )
    if r.returncode == 0:
        for line in r.stdout.splitlines():
            parts = line.split("  ", 2)
            if len(parts) >= 2:
                head[parts[1]] = parts[0]
    # Staged paths, PLUS any whose HEAD sha already disagrees with the cached row. The
    # cache reused HEAD's entry for anything not staged, which is right for a normal
    # commit and wrong after a MERGE: a merge changes files with nothing in the index,
    # so a stale row survived every regeneration. facts/efficiency.json then held a sha
    # from before 04bb05d while --write-index reported zero diff and --check-head kept
    # refusing -- the staleness test and the fix it prescribed disagreed, and the fix
    # was the wrong one (fb, 2026-09-01).
    changed = set(
        subprocess.run(
            ["git", "diff", "--cached", "--name-only", "HEAD"], cwd=root,
            capture_output=True, text=True,
        ).stdout.split()
    )
    changed |= {p for p in head if sha_head(root, p) not in (None, head[p])}
    lines = []
    for p in scoped_paths(root):
        if p in changed or p not in head:
            sha = sha_index(root, p)
            if not sha:
                continue
        else:
            sha = head[p]
        lines.append(f"{sha}  {p}  {classes.get(p, 'docs')}")
    out = os.path.join(root, "data", "pod_head_manifest.txt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return len(lines)


def read_manifest(path=None):
    """Returns path -> (sha, class). Lines without a class column default to 'docs'."""
    if path is None:
        path = MANIFEST
    m = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("  ", 2)
            if len(parts) == 3:
                sha, p, cls = parts
            else:
                sha, p = parts
                cls = "docs"
            m[p] = (sha, cls)
    return m


def check_pod(root=ROOT, scope=None):
    """Every file the manifest names must exist here and match. The pod runs this.
    scope=None checks all classes; scope='training' checks only training-class files.
    Also reports .py files no manifest entry names -- the forward check cannot see
    them, and a bare-podput script otherwise runs unregistered indefinitely.

    runs/*.jsonl drift is expected (the pod produces rows; the committed copy lags).
    It is reported, not failed -- a permanent red is no signal. Code drift FAILs."""
    manifest_path = os.path.join(root, "data", "pod_head_manifest.txt")
    if not os.path.exists(manifest_path):
        return False, f"no manifest at {os.path.relpath(manifest_path, root)}"
    manifest = read_manifest(manifest_path)
    if scope:
        manifest = {p: v for p, v in manifest.items() if v[1] == scope}
    bad = []
    runs_div = []
    for p, (want, _cls) in manifest.items():
        fp = os.path.join(root, p)
        if not os.path.exists(fp):
            # A runs/ ledger absent from the pod is not drift: pod_push skips runs/ in
            # both directions, so a ledger that only ever exists in the repo can never
            # be there. The gate treated diverged runs/ files as expected but missing
            # ones as fatal -- so adding runs/retro.jsonl turned the gate red on a file
            # nothing is allowed to push (fb, 2026-08-31).
            (runs_div if p.startswith("runs/") else bad).append(
                p if p.startswith("runs/") else f"missing {p}"
            )
        elif sha_disk(fp) != want:
            if p.startswith("runs/"):
                runs_div.append(p)
            else:
                bad.append(f"diff {p}")
    if bad:
        return False, f"{len(bad)} drifted: {'; '.join(bad)}"
    extra = unregistered_py(root, manifest)
    parts = [f"{len(manifest)} files match" + (f" (scope={scope})" if scope else "")]
    if runs_div:
        parts.append(f"{len(runs_div)} runs file(s) diverged (pod produces rows; sync to commit): {'; '.join(sorted(runs_div))}")
    if extra:
        shown = ", ".join(extra[:5]) + ("..." if len(extra) > 5 else "")
        parts.append(f"{len(extra)} UNREGISTERED .py not in manifest: {shown}")
    # A leftover in a managed dir is a deletion that never crossed, and it is NOT
    # covered above: unregistered_py sees only .py, and an extra file drifts nothing
    # the forward check can detect while still changing fp_filters.
    stale = unmanaged_extras(root, manifest)
    if stale:
        shown = ", ".join(stale[:5]) + ("..." if len(stale) > 5 else "")
        # "not on main" is what this can prove. It does NOT distinguish a deletion that
        # never crossed (pod_push only adds) from a file authored here and never
        # committed -- the first three it found were the latter. Saying "deleted on
        # main" would put a cause on the evidence and send someone hunting a commit
        # that does not exist.
        parts.append(f"{len(stale)} file(s) in managed dirs are NOT on main -- either a "
                     f"deletion that never crossed (pod_push only adds) or authored "
                     f"here and never committed: {shown}")
    return True, "; ".join(parts)


def unregistered_py(root, manifest=None):
    """.py files under root that no manifest entry names. Same scope as scoped_paths:
    EXCLUDE_DIRS and data//runs/ are not manifest territory. Reported, not failed --
    the pod legitimately holds throwaway probes -- but the output must name them."""
    if manifest is None:
        manifest = read_manifest(os.path.join(root, "data", "pod_head_manifest.txt"))
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".") and d not in EXCLUDE_DIRS]
        if os.path.relpath(dirpath, root).split(os.sep)[0] in ("data", "runs"):
            continue
        for fn in filenames:
            if fn.endswith(".py"):
                p = os.path.relpath(os.path.join(dirpath, fn), root)
                if p not in manifest:
                    out.append(p)
    return sorted(out)


#: Directories whose content is decided entirely by main -- every file in them should be
#: manifested. A file here that main does not have is not a throwaway probe; it is a
#: deletion that never crossed. pod_push only adds and overwrites, so a file main deleted
#: (filters/clean_school_math.py, 964101e) lives on the pod forever, and fp_filters
#: hashes the WHOLE directory: the fingerprint then stays wrong with no line of code
#: differing anywhere. fb's 60 deletions tonight were all this one mechanism.
MANAGED_DIRS = ("filters", "scripts/hooks", "datagen")


def unmanaged_extras(root, manifest=None):
    """Files under MANAGED_DIRS that the manifest does not name -- any extension.

    unregistered_py's counterpart for the directories where an extra file is a defect
    rather than someone's scratch script. Reported, not failed: deleting on the pod is
    the operator's call, and this says what to consider."""
    if manifest is None:
        manifest = read_manifest(os.path.join(root, "data", "pod_head_manifest.txt"))
    out = []
    for d in MANAGED_DIRS:
        base = os.path.join(root, d)
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [x for x in dirnames
                           if not x.startswith(".") and x != "__pycache__"]
            for fn in filenames:
                if fn.startswith("."):
                    continue
                p = os.path.relpath(os.path.join(dirpath, fn), root)
                if p not in manifest:
                    out.append(p)
    return sorted(out)


def plan_sync(new_path, old_path, pod_path):
    """push/del plan for `pod_push --all`. `old_path` is the pod's last manifest
    (absent -> no deletes); `pod_path` is `sha256sum` output over the new
    manifest's paths taken on the pod (missing files absent from it).
    runs/*.jsonl is skipped in both directions: the pod produces rows, so the
    sync direction there is pod -> commit, never commit -> pod."""
    new = read_manifest(new_path)
    old = read_manifest(old_path) if os.path.exists(old_path) else {}
    pod = {}
    if os.path.exists(pod_path):
        for line in open(pod_path, encoding="utf-8"):
            line = line.rstrip("\n")
            if line:
                sha, p = line.split(None, 1)
                pod[p] = sha
    plan = []
    for p, (want, _cls) in new.items():
        if p.startswith("runs/"):
            continue
        if pod.get(p) != want:
            plan.append(("push", p))
    for p in old:
        if p not in new and not p.startswith("runs/"):
            plan.append(("del", p))
    return plan


def check_head(root=ROOT):
    """The committed manifest must describe HEAD. CI runs this."""
    if not os.path.exists(MANIFEST):
        return False, "no manifest; run scripts/pod_drift.py --write"
    have = read_manifest()
    want = {}
    for p in scoped_paths(root):
        sha = sha_head(root, p)
        if sha:
            want[p] = sha
    stale = [p for p in want if have.get(p, (None,))[0] != want[p]]
    gone = [p for p in have if p not in want]
    if stale or gone:
        # Name them. A bare count makes two consecutive "1 changed" indistinguishable:
        # the same file still unfixed, or a second file that arrived with a merge. The
        # operator has to remember which, and on 2026-09-01 de read the second case as
        # the first. This list is COMPLETE, not truncated -- the counts above are its
        # length, so a reader can trust it as the whole set (fb's correction: the second
        # entry that night was new drift, not a truncated line).
        named = "; ".join(f"changed {p}" for p in sorted(stale))
        if gone:
            named += ("; " if named else "") + "; ".join(f"removed {p}" for p in sorted(gone))
        return False, (f"manifest stale: {len(stale)} changed, {len(gone)} removed; "
                       f"run --write -- {named}")
    return True, f"manifest matches HEAD ({len(want)} files)"


def is_pod(root=ROOT):
    """A git checkout has .git -- a directory in a normal clone, a FILE in a
    linked worktree. The pod's hand-pushed tree has neither. Checking isdir
    alone made every worktree look like the pod (no_ghost_running false-failed
    the first worktree commit)."""
    return not os.path.exists(os.path.join(root, ".git"))


def selftest():
    """The reverse scan on a real-shaped world: one registered script, one unregistered
    probe at root, one .py in an excluded dir and one under data/ -- only the probe is
    unregistered, and check_pod still passes while naming it.
    Also: scope filtering -- a drifted corpus-scope file must not fail --scope training,
    and a drifted training-scope file must."""
    import shutil
    import tempfile

    # Drop the caller's git env for the whole selftest, not per-subprocess: the helpers
    # below call write_manifest_index(), which runs its own git commands, and under an
    # inherited GIT_DIR those read MAIN's repo instead of the temp world -- both branches
    # then produce the same manifest and the merge case asserts "expected a conflict, got
    # a clean merge". Stripping at the entry covers every nested call.
    for _v in _GIT_ENV_VARS:
        os.environ.pop(_v, None)

    d = tempfile.mkdtemp()
    for sub in ("scripts", "datagen", "data", "workflows"):
        os.makedirs(os.path.join(d, sub), exist_ok=True)
    open(os.path.join(d, "scripts", "real.py"), "w").write("# registered\n")
    open(os.path.join(d, "probe.py"), "w").write("# throwaway\n")
    open(os.path.join(d, "workflows", "gen.py"), "w").write("# excluded dir\n")
    open(os.path.join(d, "data", "tool.py"), "w").write("# data dir\n")
    manifest = {"scripts/real.py": (sha_disk(os.path.join(d, "scripts", "real.py")), "training")}
    found = unregistered_py(d, manifest)
    assert found == ["probe.py"], found
    with open(os.path.join(d, "data", "pod_head_manifest.txt"), "w") as f:
        f.write("".join(f"{sha}  {p}  {cls}\n" for p, (sha, cls) in manifest.items()))
    ok, evidence = check_pod(d)
    assert ok and "UNREGISTERED" in evidence, evidence

    # A file main deleted but the pod still holds: pod_push only adds, so it survives
    # forever and keeps fp_filters wrong with no line of code differing. Must be named,
    # and must NOT be confused with the unregistered-probe case above.
    os.makedirs(os.path.join(d, "filters"), exist_ok=True)
    open(os.path.join(d, "filters", "clean_school_math.py"), "w").write("# deleted on main\n")
    ok_x, ev_x = check_pod(d)
    assert ok_x, ev_x
    assert "NOT on main" in ev_x and "clean_school_math.py" in ev_x, (
        f"a leftover in a managed dir was not reported: {ev_x}")
    # And a non-.py leftover, which unregistered_py cannot see at all.
    open(os.path.join(d, "filters", "stale_rules.txt"), "w").write("x\n")
    assert "stale_rules.txt" in check_pod(d)[1], "non-.py leftovers are invisible"
    os.unlink(os.path.join(d, "filters", "clean_school_math.py"))
    os.unlink(os.path.join(d, "filters", "stale_rules.txt"))

    # Scope: a corpus-scope file that drifts must not fail --scope training.
    corpus_sha = sha_disk(os.path.join(d, "scripts", "real.py"))
    manifest2 = {
        "scripts/real.py": (corpus_sha, "training"),
        "datagen/fetch_corpus.py": ("0" * 64, "corpus"),  # stale sha, corpus class
    }
    with open(os.path.join(d, "data", "pod_head_manifest.txt"), "w") as f:
        f.write("".join(f"{sha}  {p}  {cls}\n" for p, (sha, cls) in manifest2.items()))
    open(os.path.join(d, "scripts", "fetch_corpus.py"), "w").write("# corpus tool\n")
    ok_train, _ = check_pod(d, scope="training")
    assert ok_train, "corpus-scope drift must not fail --scope training"
    ok_all, _ = check_pod(d)
    assert not ok_all, "corpus-scope drift must fail the full check"
    # A training-scope file that drifts must fail --scope training.
    manifest3 = {
        "scripts/real.py": ("0" * 64, "training"),  # stale sha, training class
    }
    with open(os.path.join(d, "data", "pod_head_manifest.txt"), "w") as f:
        f.write("".join(f"{sha}  {p}  {cls}\n" for p, (sha, cls) in manifest3.items()))
    ok_train2, _ = check_pod(d, scope="training")
    assert not ok_train2, "training-scope drift must fail --scope training"

    # GIT_INDEX_FILE: `git commit B` with A also staged sets GIT_INDEX_FILE to a
    # temp index holding only B. write_manifest_index must read THAT index, so the
    # committed manifest names B and not A. (2026-08-31: the hook regenerated from
    # the shared index and swept another session's staged paths into the manifest.)
    import tempfile
    g = tempfile.mkdtemp()
    # GIT_INDEX_FILE is INHERITED by every git here, and the hook runs this selftest with
    # it already set to the real commit's temp index. Without stripping it, `git add`
    # below stages a.py/b.py into the REPO BEING COMMITTED and `git commit` here finds
    # nothing to commit, so read-tree HEAD dies on a repo with no HEAD. The selftest was
    # writing to the tree it was meant to be independent of.
    genv = _clean_git_env()
    def _g(*a):
        return subprocess.run(["git", *a], cwd=g, env=genv, capture_output=True)
    _g("init")
    _g("config", "user.email", "t@t")
    _g("config", "user.name", "t")
    open(os.path.join(g, "b.py"), "w").write("# b\n")
    _g("add", "b.py")
    _g("commit", "-m", "base")
    open(os.path.join(g, "a.py"), "w").write("# a: staged in shared index, not in this commit\n")
    _g("add", "a.py")
    tmp_index = os.path.join(g, ".git", "commit-index")
    env = dict(genv, GIT_INDEX_FILE=tmp_index)
    subprocess.run(["git", "read-tree", "HEAD"], cwd=g, env=env, capture_output=True, check=True)
    # git runs the hook with GIT_INDEX_FILE set in the environment, not as a flag.
    os.environ["GIT_INDEX_FILE"] = tmp_index
    try:
        write_manifest_index(g)
    finally:
        os.environ.pop("GIT_INDEX_FILE", None)
    m = read_manifest(os.path.join(g, "data", "pod_head_manifest.txt"))
    assert "a.py" not in m, "manifest named a.py, absent from the commit's index"
    assert "b.py" in m, "manifest dropped b.py, present in the commit's index"

    # Merge-commit regeneration: when the hook runs on a merge commit, the index
    # it reads IS the merged index. The regenerated manifest must name every file
    # from BOTH parents with the merged shas -- taking either side's manifest
    # loses the other branch's files (2026-08-31 worktree ruling). Two-branch
    # world: main adds c.py, branch de adds b.py; both regenerated the manifest,
    # so the merge conflicts on it and the hook regenerates from the merged index.
    import tempfile
    h = tempfile.mkdtemp()

    henv = _clean_git_env()

    def g(*args, check=True):
        return subprocess.run(["git", *args], cwd=h, env=henv, capture_output=True,
                              text=True, check=check)

    g("init")
    g("checkout", "-b", "main")
    g("config", "user.email", "t@t")
    g("config", "user.name", "t")
    open(os.path.join(h, "a.py"), "w").write("# a\n")
    g("add", "a.py")
    write_manifest_index(h)  # the hook, on main's first commit
    g("add", "data/pod_head_manifest.txt")
    g("commit", "-m", "a")
    g("checkout", "-b", "de")
    open(os.path.join(h, "b.py"), "w").write("# b\n")
    g("add", "b.py")
    write_manifest_index(h)
    g("add", "data/pod_head_manifest.txt")
    g("commit", "-m", "b")
    g("checkout", "main")
    open(os.path.join(h, "c.py"), "w").write("# c\n")
    g("add", "c.py")
    write_manifest_index(h)
    g("add", "data/pod_head_manifest.txt")
    g("commit", "-m", "c")
    r = g("merge", "de", "--no-edit", check=False)
    assert r.returncode != 0, "expected a manifest conflict, got a clean merge"
    # The hook on the conflict-resolution commit: regenerate from the merged index.
    write_manifest_index(h)
    m = read_manifest(os.path.join(h, "data", "pod_head_manifest.txt"))
    assert {"a.py", "b.py", "c.py"} <= set(m), f"merged manifest lost files: {sorted(m)}"
    for p in ("a.py", "b.py", "c.py"):
        assert m[p][0] == sha_disk(os.path.join(h, p)), f"{p} sha is not the merged sha"
    # plan_sync: a matching file is untouched, a changed file is pushed, a
    # manifest-left file is deleted, runs/ is skipped both ways.
    import tempfile
    j = tempfile.mkdtemp()
    with open(os.path.join(j, "new"), "w") as f:
        f.write(f"{'a' * 64}  a.py  training\n{'b' * 64}  b.py  docs\n{'c' * 64}  runs/x.jsonl  docs\n")
    with open(os.path.join(j, "old"), "w") as f:
        f.write(f"{'a' * 64}  a.py  training\n{'d' * 64}  d.py  docs\n")
    with open(os.path.join(j, "pod"), "w") as f:
        f.write(f"{'a' * 64}  a.py\n{'0' * 64}  b.py\n")
    plan = plan_sync(os.path.join(j, "new"), os.path.join(j, "old"), os.path.join(j, "pod"))
    assert plan == [("push", "b.py"), ("del", "d.py")], plan
    # A runs/ ledger the pod has never seen is not drift: pod_push skips runs/, so
    # a repo-only ledger can never exist there. Regression for the red the new
    # runs/retro.jsonl produced (2026-08-31).
    k = tempfile.mkdtemp()
    os.makedirs(os.path.join(k, "data"), exist_ok=True)
    os.makedirs(os.path.join(k, "runs"), exist_ok=True)
    real = os.path.join(k, "kept.py")
    with open(real, "w") as f:
        f.write("x = 1\n")
    with open(os.path.join(k, "data", "pod_head_manifest.txt"), "w") as f:
        f.write(f"{sha_disk(real)}  kept.py  training\n{'e' * 64}  runs/retro.jsonl  docs\n")
    ok, ev = check_pod(k)
    assert ok, f"a runs/ ledger missing on the pod must not fail the gate: {ev}"
    assert "retro" in ev, f"the absent ledger must still be reported: {ev}"
    # ...while a missing CODE file still fails.
    with open(os.path.join(k, "data", "pod_head_manifest.txt"), "a") as f:
        f.write(f"{'f' * 64}  gone.py  training\n")
    ok, ev = check_pod(k)
    assert not ok and "gone.py" in ev, f"a missing code file must still fail: {ok} {ev}"

    # check_head names the files, because a bare count cannot distinguish "the same file
    # is still unfixed" from "a merge brought a second one". de hit exactly that on
    # 2026-09-01: two consecutive "1 changed" lines, and the second was new drift.
    #
    # Built from HEAD's REAL manifest with one entry's sha corrupted, and it reads the
    # real repo directly -- check_head only needs MANIFEST and `git show HEAD:path`, so
    # no copy is required. My first version copied .git with copytree and skipped the
    # whole block under `if m`, because in a WORKTREE .git is a FILE, not a directory:
    # NotADirectoryError, silently swallowed, assertion never evaluated. It passed on the
    # count-only code it exists to catch -- the same vacuous shape, in the test written to
    # prevent it (de, 2026-09-01).
    m = tempfile.mkdtemp()
    _orig_manifest = MANIFEST
    try:
        rows = open(_orig_manifest, encoding="utf-8").read().splitlines()
        victim, out = None, []
        for r in rows:
            parts = r.split()
            if victim is None and len(parts) >= 2 and parts[1].endswith(".py"):
                victim = parts[1]
                out.append(f"{'0' * 64}  {'  '.join(parts[1:])}")
            else:
                out.append(r)
        assert victim, "HEAD's manifest names no .py -- this fixture cannot corrupt one"
        broken = os.path.join(m, "pod_head_manifest.txt")
        with open(broken, "w") as f:
            f.write("\n".join(out) + "\n")
        globals()["MANIFEST"] = broken
        ok, ev = check_head(ROOT)
        assert not ok, f"a corrupted manifest entry must fail check_head: {ev}"
        assert victim in ev, (
            f"check_head must NAME the stale file, not only count it -- an operator "
            f"seeing two '1 changed' lines cannot tell a new file from an unfixed one. "
            f"Expected {victim} in: {ev}")
    finally:
        globals()["MANIFEST"] = _orig_manifest
        shutil.rmtree(m, ignore_errors=True)

    print("pod_drift selftest OK:", evidence)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "--check"
    if mode == "--write":
        n = write_manifest()
        print(f"wrote {n} entries to {os.path.relpath(MANIFEST, ROOT)}")
    elif mode == "--write-index":
        n = write_manifest_index()
        print(f"wrote {n} entries from index to {os.path.relpath(MANIFEST, ROOT)}")
    elif mode == "--selftest":
        selftest()
    elif mode == "--list-scoped":
        print("\n".join(scoped_paths()))
    elif mode == "--check":
        scope = None
        if "--scope" in sys.argv:
            scope = sys.argv[sys.argv.index("--scope") + 1]
        if is_pod(ROOT):
            ok, evidence = check_pod(ROOT, scope=scope)
        elif os.environ.get("CI") == "true":
            ok, evidence = check_head(ROOT)
        else:
            print("dev checkout: nothing to check (CI gates manifest freshness, the pod gates file drift)")
            sys.exit(0)
        print(("OK: " if ok else "DRIFT: ") + evidence)
        sys.exit(0 if ok else 1)
    elif mode == "--plan-sync":
        # args: <old manifest from pod> <pod sha256sum output>; prints "push <p>" /
        # "del <p>" lines for pod_push --all.
        for op, p in plan_sync(MANIFEST, sys.argv[2], sys.argv[3]):
            print(f"{op} {p}")
    elif mode == "--check-head":
        ok, evidence = check_head(ROOT)
        print(("OK: " if ok else "STALE: ") + evidence)
        sys.exit(0 if ok else 1)
    else:
        print(__doc__)
        sys.exit(2)


if __name__ == "__main__":
    main()
