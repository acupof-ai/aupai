#!/usr/bin/env python3
"""A ledger writer refuses in the integration tree, and nowhere else.

4c's ruling 2026-09-05, after two rows landed in the integration tree ten minutes apart: b0's
task row, then 44's board row. AGENTS.md already said to run these in your worktree, and the
rule's coverage row explained why prose was all it could be -- "the invoking directory is a
shell fact no artifact records". It is recoverable from the tree the writer is about to append
to, which is what this makes checkable.

THE PREDICATE IS NOT THE BRANCH, AND THIS FILE'S FIRST VERSION GOT THAT WRONG. It tested
`branch == "main"` and its worlds asserted branch semantics; hours later the integration tree
was detached on purpose (tilerl's flip, main 0425accb) and all three guards silently turned
OFF in the one tree they exist for. The predicate now lives in scripts/integration_tree.py --
main worktree of a common git dir that HAS linked worktrees -- and these worlds BUILD that
property with `git worktree add` instead of naming a branch.

tilerl found the same defect in the hook's own selftest from the other side: worlds 1-4 there
built a bare `git init`, which is not an integration tree by this definition, so world 1 was
asserting "a commit here is refused" against a repo that never qualified. It passed for years
because the branch test did not care. A world that does not hold the property proves nothing
about a guard that reads it.

WHAT THE REFUSAL PREVENTS is not the row -- the row is valid content -- it is the DIRTY LEDGER.
The integration tree's pre-commit hook refuses the commit, so the append sits uncommitted in the
tree every other session merges through, and the next merge aborts on it. That is why the guard
is at the write and not at the commit: by the time the hook speaks, the file is already dirty.

THE FAIL-OPEN WORLDS ARE LOAD-BEARING, and they are the half someone tightening the guard would
drop. Where git cannot answer -- no repository, no git binary -- the tree is not the integration
tree, and refusing would break every one of harness.py's 27 mkdtemp worlds and every CI runner.
A guard whose broken state blocks the write is a guard someone deletes. An UNIMPORTABLE
predicate is the one case that is loud instead of silent: with the tree detached there is no
branch test left to fall back to, so a swallowed ImportError would leave the integration tree
wholly unguarded rather than merely degraded.

AND THE WRITERS MUST CALL IT. A guard nothing calls satisfies every world above (§233), so the
last worlds read the three call sites: harness._append_task (tasks.jsonl and friction.jsonl both
reach disk through it), harness._write_tasks (a rewrite, which dirties the whole file rather
than one line), and board.append (the third ledger, and the one 44 wrote). board.py imports the
guard from harness rather than copying it -- a second implementation of one rule is how
FRICTION_KINDS came to reject a kind this repo's own merge_main.sh emits.

W11-W12 ENUMERATE, THEY DO NOT LIST. The guard's population is every ledger writer in the tree,
and the population is read from the filesystem at test time -- `git grep` over tracked files for the writers,
`.gitattributes` for the union ledgers -- never from a literal list of ledgers (4c's ruling,
de-98: "my 4 was itself a list -- the three harness writers plus the one that had just bitten
us"). A new session ledger or a new unguarded writer turns W11 red by itself; the only list
this file carries is the adjudicated exclusions, each a run-side or one-off writer with a
reason, and adding a line there is a conscious adjudication in a commit -- the registration
event. Fixture builders (selftest/_broken/_world worlds and test_*.py) write into tmp worlds,
never the repo tree, and are skipped by convention; a fixture that wrote the real tree would
have to be adjudicated too, which is the point.

restartable: yes -- every world is a fresh temp git repo removed in a finally. Nothing reads or
writes the repository's real ledgers.
"""
import ast
import inspect
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "datagen"))


_CLEAN_ENV = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
_CLEAN_ENV.update(GIT_CONFIG_GLOBAL="/dev/null", GIT_CONFIG_NOSYSTEM="1")


def _git(d, *a):
    """git in a throwaway world, with the caller's git environment stripped.

    This had no env= at all while running `init`, `config` and `branch -M main`. Under a
    leaked GIT_DIR those write to the SHARED repository, and with GIT_DIR pointing at a
    worktree gitdir `git init` flips core.bare and takes every session's git down (twice on
    2026-09-02). `branch -M` also rewrites a [branch] section, which the pre-commit
    shared-repo guard deliberately excludes from its digest -- so without this strip the
    guard would be silent on exactly that write. Pattern from
    test_behind_main_overlap.py:55-56."""
    return subprocess.run(["git", "-C", d, *a], capture_output=True, text=True, timeout=60,
                          env=_CLEAN_ENV)


def _integration_tree(parent):
    """A tree that IS the integration tree: the main worktree, with a linked worktree hanging
    off it. Built, not asserted -- that is the whole lesson of this file's first version."""
    d = tempfile.mkdtemp(dir=parent)
    _git(d, "init", "-q", ".")
    _git(d, "config", "user.email", "t@example.invalid")
    _git(d, "config", "user.name", "t")
    _git(d, "commit", "-q", "--allow-empty", "-m", "base")
    _git(d, "branch", "-M", "main")
    wt = tempfile.mkdtemp(dir=parent)
    os.rmdir(wt)  # git worktree add wants the path absent
    r = _git(d, "worktree", "add", "-q", "-b", "sidebranch", wt)
    os.makedirs(os.path.join(d, "runs"), exist_ok=True)
    os.makedirs(os.path.join(wt, "runs"), exist_ok=True)
    return d, os.path.join(d, "runs", "tasks.jsonl"), wt, os.path.join(wt, "runs", "tasks.jsonl"), r


def _report(fails):
    if fails:
        for f in fails:
            print(f"  FAIL {f}")
        print(f"\n{len(fails)} failure(s)")
        return 1
    print("  ledger writers: refuse in the integration tree (detached too), not in a linked "
          "worktree, fail open on no-git/no-git-binary; W11 enumerates every writer from the "
          "filesystem and W12 the union population from .gitattributes")
    return 0


# --- W11: the writer enumeration ---------------------------------------------------------
#
# Every tracked .py that opens a runs/*.jsonl ledger for writing, with the enclosing function
# and whether it calls the guard. AST, not grep: a grep for 'runs/' cannot tell a write from a
# read and cannot resolve a module constant. The path resolver is ROOT-independent on purpose
# -- every writer builds its path as os.path.join(ROOT, "runs", "x.jsonl"), and ROOT is
# unresolvable statically; the runs/<...>.jsonl TAIL is what identifies the ledger.

_WRITE_FLAGS = {"O_WRONLY", "O_RDWR", "O_APPEND", "O_CREAT"}
_TMP_FUNCS = ("mkdtemp", "TemporaryDirectory", "_tmp_repo", "tempdir")
_FIXTURE_PREFIXES = ("selftest", "_selftest", "_broken", "_world", "world", "_demo", "_two",
                     "_bad", "_fix", "_board_event", "_in_subprocess", "_repo", "retract_cases",
                     "_fixture", "_event")

# Adjudicated: run-side and one-off writers that are NOT session ledger writers. Each line is a
# conscious decision; a new writer turns W11 red until it is guarded or adjudicated here.
_EXCLUDED_WRITERS = {
    ("scripts/memory_diag.py", "log_diag"): "train-side diag writer, frozen for p500m_20b_0902; "
                                            "runs on the pod, which is not a git repo, so the guard fails open there",
    ("scripts/moe_diag.py", "log_diag"): "same as memory_diag",
    ("scripts/eval_artifacts.py", "attest"): "eval-side; runs/artifact_refs.jsonl is written by eval runs on the pod",
    ("scripts/b0_sd_cu_rescore.py", "main"): "one-off rescore tool, ran once",
    ("scripts/e1_39_close_c6e.py", "<module>"): "one-off task-close script (e1-39)",
    ("scripts/fable5_audit_sample.py", "main"): "one-off audit sample",
    ("scripts/write_prereg_moe48_30b.py", "main"): "one-off prereg writer, ran once",
}


def _ledger_tail(node, consts):
    """The runs/<...>.jsonl tail an expression statically names, else None. ROOT and friends
    are deliberately unresolvable -- the tail is the identity."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        i = node.value.rfind("runs/")
        if i >= 0 and node.value.endswith(".jsonl"):
            return node.value[i:]
        return None
    if isinstance(node, ast.Name):
        return consts.get(node.id)
    if isinstance(node, ast.JoinedStr):
        parts = [v.value for v in node.values
                 if isinstance(v, ast.Constant) and isinstance(v.value, str)]
        if len(parts) != len(node.values):
            return None
        return _ledger_tail(ast.Constant("".join(parts)), consts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _ledger_tail(node.right, consts) or _ledger_tail(node.left, consts)
    if isinstance(node, ast.BoolOp):
        for v in node.values:
            t = _ledger_tail(v, consts)
            if t:
                return t
        return None
    if isinstance(node, ast.IfExp):
        return _ledger_tail(node.orelse, consts) or _ledger_tail(node.body, consts)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
            and node.func.attr == "join":
        for a in node.args:
            t = _ledger_tail(a, consts)
            if t and t.startswith("runs/"):
                return t
        for i, a in enumerate(node.args):
            if isinstance(a, ast.Constant) and a.value == "runs":
                tail = [b.value for b in node.args[i + 1:]
                        if isinstance(b, ast.Constant) and isinstance(b.value, str)]
                if len(tail) == len(node.args[i + 1:]) and tail and tail[-1].endswith(".jsonl"):
                    return "runs/" + "/".join(tail)
    return None


def _is_tmp(node, consts, tmp):
    """A tmp-world root: mkdtemp/_tmp_repo call, a constant containing 'tmp', or a Name so marked."""
    if isinstance(node, ast.Name):
        return node.id in tmp or consts.get(node.id) == "__TMP__"
    if isinstance(node, ast.Call):
        f = node.func
        name = f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else "")
        if name in _TMP_FUNCS:
            return True
        if name == "join":
            return any(_is_tmp(a, consts, tmp) for a in node.args)
    if isinstance(node, ast.Constant) and isinstance(node.value, str) and "tmp" in node.value:
        return True
    if isinstance(node, ast.BoolOp):
        return any(_is_tmp(v, consts, tmp) for v in node.values)
    return False


def _write_mode(call):
    """open(): the mode string. os.open(): the flag bits. Default open mode is read."""
    f = call.func
    if isinstance(f, ast.Name) and f.id == "open":
        mode = None
        if len(call.args) >= 2 and isinstance(call.args[1], ast.Constant) \
                and isinstance(call.args[1].value, str):
            mode = call.args[1].value
        for kw in call.keywords:
            if kw.arg == "mode" and isinstance(kw.value, ast.Constant) \
                    and isinstance(kw.value.value, str):
                mode = kw.value.value
        return mode is not None and (mode.lstrip("tb").startswith(("w", "a", "x")) or "+" in mode)
    if isinstance(f, ast.Attribute) and f.attr == "open" and isinstance(f.value, ast.Name) \
            and f.value.id == "os" and len(call.args) >= 2:
        return any(isinstance(n, (ast.Name, ast.Attribute))
                   and getattr(n, "id", getattr(n, "attr", "")) in _WRITE_FLAGS
                   for n in ast.walk(call.args[1]))
    return False


def ledger_writers(root):
    """Every (relpath, lineno, ledger, func, guarded) write site in tracked .py files."""
    # git grep pre-filters in C: reading every tracked .py to string-check it cost ~1s and
    # pushed this check past its deadline. A file naming neither string cannot hold a site.
    def mentioning(needle):
        r = subprocess.run(["git", "-C", root, "grep", "-l", needle, "--", "*.py"],
                           capture_output=True, text=True, timeout=60)
        return set(r.stdout.split()) if r.returncode == 0 else set()

    files = sorted(mentioning("runs/") & mentioning(".jsonl"))
    out = []
    for rel in files:
        path = os.path.join(root, rel)
        try:
            src = open(path, encoding="utf-8").read()
            tree = ast.parse(src)
        except (OSError, SyntaxError):
            continue
        mod = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                    and isinstance(node.targets[0], ast.Name):
                t = _ledger_tail(node.value, mod)
                if t:
                    mod[node.targets[0].id] = t

        class V(ast.NodeVisitor):
            def __init__(self):
                self.stack = []
                self.locals = [dict(mod)]
                self.tmp = [set()]

            def visit_FunctionDef(self, node):
                self.stack.append(node)
                self.locals.append(dict(mod))
                self.tmp.append(set())
                for stmt in ast.walk(node):
                    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 \
                            and isinstance(stmt.targets[0], ast.Name):
                        t = _ledger_tail(stmt.value, self.locals[-1])
                        if t:
                            self.locals[-1][stmt.targets[0].id] = t
                        if _is_tmp(stmt.value, self.locals[-1], self.tmp[-1]):
                            self.tmp[-1].add(stmt.targets[0].id)
                self.generic_visit(node)
                self.tmp.pop()
                self.locals.pop()
                self.stack.pop()

            visit_AsyncFunctionDef = visit_FunctionDef

            def visit_Call(self, node):
                f = node.func
                is_open = (isinstance(f, ast.Name) and f.id == "open") or (
                    isinstance(f, ast.Attribute) and f.attr == "open"
                    and isinstance(f.value, ast.Name) and f.value.id == "os")
                if is_open and node.args and _write_mode(node):
                    target = _ledger_tail(node.args[0], self.locals[-1])
                    if target and not _is_tmp(node.args[0], self.locals[-1], self.tmp[-1]):
                        fn = self.stack[-1] if self.stack else None
                        fname = fn.name if fn else "<module>"
                        body = ast.get_source_segment(src, fn) if fn else ""
                        guarded = ("refuse_in_integration_tree" in body) or ("append_ledger" in body)
                        out.append((rel, node.lineno, target, fname, guarded))
                self.generic_visit(node)

        V().visit(tree)
    return out


def _is_fixture(rel, fname):
    """Fixture builders write tmp worlds, never the repo tree (see the W11-W12 docstring)."""
    return os.path.basename(rel).startswith("test_") or fname.startswith(_FIXTURE_PREFIXES)


def main():
    import harness

    fails = []
    tmp = tempfile.mkdtemp(prefix="itree_")
    try:
        d_int, p_int, d_wt, p_wt, r_add = _integration_tree(tmp)
        if r_add.returncode != 0:
            return _report([f"the world could not be built: `git worktree add` failed "
                            f"({r_add.stderr.strip()[:120]}) -- a world that does not hold the "
                            f"property proves nothing about a guard that reads it"])

        # W1: the integration tree REFUSES. The whole point.
        if not harness.refuse_in_integration_tree("w1", path=p_int):
            fails.append("W1: the integration tree did not refuse -- this is the tree whose "
                         "pre-commit hook cannot commit the row, so the append would sit dirty "
                         "in the tree every session merges through")

        # W2: THE SAME TREE, DETACHED, still refuses. The world tilerl's flip created, and the
        # one that turned all three guards off when the predicate read the branch. Asserts the
        # tree really is detached, so it cannot pass for the wrong reason.
        _git(d_int, "checkout", "-q", "--detach", "HEAD")
        sr = _git(d_int, "symbolic-ref", "--short", "HEAD")
        if sr.returncode == 0 and sr.stdout.strip() == "main":
            fails.append("W2 precondition: the tree is not actually detached, so this world says "
                         "nothing about the flip")
        if not harness.refuse_in_integration_tree("w2", path=p_int):
            fails.append("W2: a DETACHED integration tree did not refuse -- this is exactly the "
                         "state the flip created (main 0425accb), where a branch-name predicate "
                         "reads 'HEAD' and every guard silently turns off")

        # W3: a LINKED WORKTREE of the same repo does not refuse. Every session's normal write,
        # and the direction a too-broad guard breaks -- it would block all of them.
        if harness.refuse_in_integration_tree("w3", path=p_wt):
            fails.append("W3: a linked worktree refused -- that is every session's normal write")

        # W4: a standalone repo does not refuse. Not decoration: without the has-linked-worktrees
        # clause, a plain clone is its own main worktree, so CI and all 27 of harness.py's
        # git-init fixtures would refuse every write.
        d_solo = tempfile.mkdtemp(dir=tmp)
        _git(d_solo, "init", "-q", ".")
        _git(d_solo, "config", "user.email", "t@example.invalid")
        _git(d_solo, "config", "user.name", "t")
        _git(d_solo, "commit", "-q", "--allow-empty", "-m", "base")
        _git(d_solo, "branch", "-M", "main")
        os.makedirs(os.path.join(d_solo, "runs"), exist_ok=True)
        if harness.refuse_in_integration_tree("w4", path=os.path.join(d_solo, "runs", "tasks.jsonl")):
            fails.append("W4: a standalone clone on main refused -- it is its own main worktree "
                         "but nothing hangs off it, so it blocks nobody; CI is this shape")

        # W5: no repository at all -> fail open.
        d_bare = tempfile.mkdtemp(dir=tmp)
        os.makedirs(os.path.join(d_bare, "runs"), exist_ok=True)
        if harness.refuse_in_integration_tree("w5", path=os.path.join(d_bare, "runs", "tasks.jsonl")):
            fails.append("W5: a non-repository refused -- every _tmp_repo() world is this shape "
                         "and the selftest would refuse to write its own fixtures")

        # W6: git itself unavailable -> fail open. PATH emptied, so the subprocess raises rather
        # than returning nonzero: a different code path from W5.
        saved_path = os.environ.get("PATH", "")
        try:
            empty = os.path.join(tmp, "empty-bin")
            os.makedirs(empty, exist_ok=True)
            os.environ["PATH"] = empty
            refused = harness.refuse_in_integration_tree("w6", path=p_int)
        finally:
            os.environ["PATH"] = saved_path
        if refused:
            fails.append("W6: refused when git was unavailable -- an unanswerable predicate must "
                         "not block the write")

        # W7-W9: THE THREE WRITERS CALL IT. A guard nobody calls passes W1-W6 (§233).
        for fn in (harness._append_task, harness._write_tasks):
            if "refuse_in_integration_tree" not in inspect.getsource(fn):
                fails.append(f"W7: harness.{fn.__name__} does not call the guard -- both ledgers "
                             f"reach disk through these two functions, so a guard in the CLI ops "
                             f"instead would miss the next op someone adds")
        bsrc = open(os.path.join(ROOT, "scripts", "board.py"), encoding="utf-8").read()
        i = bsrc.find("def append(")
        if i < 0:
            fails.append("W9: scripts/board.py has no append() -- the writer moved")
        elif "refuse_in_integration_tree" not in bsrc[i:i + 2000]:
            fails.append("W9: board.append does not call the guard -- board.jsonl is the third "
                         "ledger and the one 44 wrote into the integration tree")
        elif "from harness import" not in bsrc[i:i + 2000]:
            fails.append("W9: board.append does not IMPORT the guard -- a local copy is a second "
                         "implementation of one rule, which is how FRICTION_KINDS came to reject "
                         "a kind merge_main.sh emits")

        # W10: the predicate is the SHARED one, not a private copy. tilerl's hook calls the same
        # function; two spellings of a similar idea is the defect this file exists downstream of.
        #
        # READ THE BODY, NOT THE WHOLE SOURCE. The first version grepped the function's source
        # for "symbolic-ref" and went red on its own docstring, which explains why that predicate
        # was abandoned -- a substring test cannot tell code from the prose recording why the
        # code is not there any more. Same shape as de-55's "signalled" grep, one hour apart.
        gsrc = inspect.getsource(harness.refuse_in_integration_tree)
        body = gsrc.split('"""')[2] if gsrc.count('"""') >= 2 else gsrc
        if "from integration_tree import is_integration_tree" not in body:
            fails.append("W10: the guard does not import integration_tree.is_integration_tree -- "
                         "the hook and the writers must read one implementation, or they drift")
        if "symbolic-ref" in body or "abbrev-ref" in body:
            fails.append("W10: the guard's BODY still reads a branch name -- that predicate went "
                         "inert the moment the integration tree was detached (2026-09-05)")

        # W11: EVERY SESSION LEDGER WRITER CALLS THE GUARD. The population is the filesystem:
        # git grep over tracked files at test time, so a new unguarded writer turns this red by
        # itself. The only list is the adjudicated exclusions (run-side and one-off writers,
        # each with a reason in _EXCLUDED_WRITERS); fixtures write tmp worlds and are skipped
        # by convention.
        writers = ledger_writers(ROOT)
        for rel, lineno, target, fname, guarded in writers:
            if guarded or _is_fixture(rel, fname) or (rel, fname) in _EXCLUDED_WRITERS:
                continue
            fails.append(f"W11: {rel}:{lineno} opens {target} for writing in {fname}() without "
                         f"calling the guard -- a new session ledger writer, or one that forgot "
                         f"refuse_in_integration_tree/append_ledger. Guard it, or adjudicate it "
                         f"in _EXCLUDED_WRITERS with a reason")

        # W12: THE UNION LEDGER POPULATION IS ENUMERATED, NEVER LISTED. Every merge=union path
        # in .gitattributes either has a guarded writer (W11's scan) or is covered by the
        # generic `harness ledger append` writer, which accepts any runs/*.jsonl path. A new
        # union ledger outside that shape -- or the generic writer going away -- turns this red.
        ga = open(os.path.join(ROOT, ".gitattributes"), encoding="utf-8").read()
        union_ledgers = [ln.split()[0] for ln in ga.splitlines()
                         if ln.strip() and "merge=union" in ln and ln.split()[0].endswith(".jsonl")]
        guarded_targets = {t for (_r, _l, t, _f, g) in writers if g}
        for led in union_ledgers:
            if led in guarded_targets:
                continue
            if not (led.startswith("runs/") and led.endswith(".jsonl")):
                fails.append(f"W12: union ledger {led!r} has no guarded writer and is outside "
                             f"runs/*.jsonl, so the generic writer cannot cover it -- give it a "
                             f"guarded writer or move it under runs/")
        for name in ("cmd_review", "cmd_ledger_append"):
            fn = getattr(harness, name, None)
            if fn is None:
                fails.append(f"W12: harness.{name} is gone -- the generic guarded writers for "
                             f"review.jsonl and the writerless ledgers (retro, ledger_resolutions)")
            elif "append_ledger" not in inspect.getsource(fn):
                fails.append(f"W12: harness.{name} no longer calls append_ledger -- the writer "
                             f"lost its guard")
        if harness.cmd_ledger_append(["--path", "../evil.jsonl", "--row", "{}"]) != 1:
            fails.append("W12: `harness ledger append` accepted a path outside runs/ -- the "
                         "generic writer's trust boundary is gone")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    return _report(fails)


if __name__ == "__main__":
    sys.exit(main())
