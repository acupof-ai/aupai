#!/usr/bin/env python3
"""Every eval artifact path carries the checkpoint that produced it (de-24).

The defect (fb, 2026-09-02). score_matrix scored two checkpoints in one session and the
second came back ArtifactExists instead of a number, because eval/l1_fewshot.py wrote
`data/eval/preds_l1_d3.jsonl` -- a path identical for every checkpoint. open_artifact was
right to refuse; the path was wrong. eval/code_fewshot.py carried the same defect and
nobody had hit it yet, which is the part a per-incident fix misses.

The property, checked over the source rather than by running five GPU evals: in every
file that calls open_artifact, the expression it passes must interpolate the checkpoint.
Three files already did it (math_zh, code_zh, math_hard) and two did not, so this is a
rule the codebase mostly follows and had no way to state.

Why the checkpoint name and not --run: --run also versions the path, and it is what the
facts' provenance uses. But --run is a flag a person remembers to pass, and score_matrix
does not pass it (eval/score_matrix.py:197 builds `--ckpt X --out Y` and nothing else).
An artifact path that is only unique when someone passes an optional flag is unique by
courtesy. The checkpoint is in scope at every one of these call sites already.

AST, not regex. The first version matched `open_artifact(` textually and read the nearest
assignment above it with a regex, which found four things that are not call sites at all
(prose in scripts/harness.py's docstrings, and this file's own selftest strings) and
truncated every multi-line path expression at the first line. A syntax tree cannot
mistake a docstring for a call, and ast.unparse returns the whole expression however it
was wrapped.

    python3 scripts/test_artifact_path_ckpt.py --selftest

# restartable: reads source files, writes nothing.
"""

import ast
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# What counts as naming the checkpoint: the argparse value under either spelling the repo
# uses (`args.ckpt` in eval/, `a.ckpt` in math_hard.py), or score_matrix's `ckpt_name`.
CKPT_NAMES = {"ckpt_name", "ckpt_path"}


def _names_ckpt(node):
    """Does this expression mention the checkpoint anywhere inside it?"""
    for n in ast.walk(node):
        if isinstance(n, ast.Attribute) and n.attr == "ckpt":
            return True
        if isinstance(n, ast.Name) and n.id in CKPT_NAMES:
            return True
    return False


def artifact_path_exprs(src):
    """[(var, expr_or_None)] for every open_artifact(<var>, ...) in this source.

    The path is built a few lines above the call at every site, so the expression that
    matters is the assignment, not the argument. A literal argument is returned as its own
    expression; an argument whose assignment is not in this file yields None, which the
    caller reports rather than skips -- an unresolvable path is not a passing one.

    Calls inside a selftest function are skipped, and that exemption is the honest form of
    what a file allow-list was doing by hand. MEASURED: every open_artifact call in the
    repo that legitimately has no checkpoint in scope is inside one -- nine in
    scripts/eval_artifacts.py `_selftest` and one in scripts/harness.py
    `_selftest_attest_written_path`. A selftest exercises the refusal on temp paths; it is
    not writing a checkpoint's predictions. Naming the property beats naming the two files
    that happen to have it today.
    """
    tree = ast.parse(src)
    assigns = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name):
                    assigns[t.id] = n.value
    exempt = set()
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) and "selftest" in fn.name:
            for n in ast.walk(fn):
                if isinstance(n, ast.Call):
                    exempt.add(id(n))
    out = []
    for n in ast.walk(tree):
        if not (isinstance(n, ast.Call) and getattr(n.func, "id", None) == "open_artifact"):
            continue
        if not n.args or id(n) in exempt:
            continue
        arg = n.args[0]
        if isinstance(arg, ast.Name):
            out.append((arg.id, assigns.get(arg.id)))
        else:
            out.append((ast.unparse(arg)[:40], arg))
    return out


def check(root=ROOT):
    """(violations, checked_files). A violation is (path, var, why)."""
    bad, files = [], []
    for d in ("eval", "probes", "scripts", "datagen"):
        dd = os.path.join(root, d)
        if not os.path.isdir(dd):
            continue
        for b in sorted(os.listdir(dd)):
            if not b.endswith(".py"):
                continue
            rel = os.path.join(d, b)
            src = open(os.path.join(dd, b), encoding="utf-8").read()
            if "open_artifact(" not in src:
                continue
            try:
                exprs = artifact_path_exprs(src)
            except SyntaxError as e:
                bad.append((rel, "-", f"does not parse: {e}"))
                continue
            if not exprs:
                continue
            files.append(rel)
            for var, expr in exprs:
                if expr is None:
                    bad.append((rel, var, "the path variable is not assigned in this file, "
                                          "so what it holds cannot be read here"))
                elif not _names_ckpt(expr):
                    bad.append((rel, var, "the path does not interpolate the checkpoint, so "
                                          "every checkpoint writes the same file and the "
                                          "second one is refused"))
    return bad, files


def main():
    bad, files = check()
    if not files:
        print("FAIL: no file calls open_artifact -- this check has nothing to look at, "
              "which means the artifact discipline moved and this test did not follow")
        return 1
    if bad:
        print(f"FAIL: {len(bad)} eval artifact path(s) do not name their checkpoint")
        for path, var, why in bad:
            print(f"  {path}: {var} -- {why}")
        print("\nAdd os.path.basename(args.ckpt) to the path, as eval/math_zh.py:103 and "
              "eval/code_zh.py:160 do.")
        return 1
    print(f"OK: {len(files)} file(s) write eval artifacts, every path names its checkpoint")
    return 0


def _selftest():
    """The check must FAIL on the real pre-fix expression and PASS on the real fixed one.

    Both strings are copied from git, not invented: a hand-written 'bad path' would share
    this test's own idea of what bad looks like, which is how a guard passes every case its
    author imagined and none that occurred (AGENTS.md: broken worlds mutate a real
    artifact).
    """
    pre_fix = (
        'def main():\n'
        '    preds_path = os.path.join(ROOT, "data", "eval",\n'
        '                              f"preds_l1_d{args.demos}"\n'
        '                              + (f".t{args.temperature}" if args.temperature else "")\n'
        '                              + ".jsonl")\n'
        '    with open_artifact(preds_path, force=args.force, run=args.run) as fout:\n'
        '        pass\n'
    )
    post_fix = pre_fix.replace('f"preds_l1_d{args.demos}"',
                               'f"preds_l1_d{args.demos}_{os.path.basename(args.ckpt)}"')
    for label, src, want_bad in (("pre-fix", pre_fix, True), ("post-fix", post_fix, False)):
        exprs = artifact_path_exprs(src)
        assert len(exprs) == 1, f"{label}: found {len(exprs)} artifact paths, expected 1"
        var, expr = exprs[0]
        assert var == "preds_path", f"{label}: read the variable as {var!r}"
        assert expr is not None, f"{label}: the assignment was not found"
        # The whole expression, not its first line: the pre-fix path is three lines and a
        # truncating reader would judge only the first.
        assert ".jsonl" in ast.unparse(expr), f"{label}: the expression was truncated"
        got_bad = not _names_ckpt(expr)
        assert got_bad is want_bad, (
            f"{label}: the checkpoint-in-path predicate returned {not got_bad} on\n"
            f"{ast.unparse(expr)}")

    # A docstring that merely SAYS open_artifact must not register as a call site: four of
    # the first version's six findings were prose.
    prose = '"""see open_artifact(path, run=...) for versioning."""\nx = 1\n'
    assert artifact_path_exprs(prose) == [], "a docstring mention read as a call site"

    # And the live tree, which is the case that matters: the two files fb's report named
    # must now pass, and the predicate must be non-vacuous.
    bad, files = check()
    assert files, "no file calls open_artifact"
    for want in ("l1_fewshot.py", "code_fewshot.py"):
        assert any(f.endswith(want) for f in files), (
            f"eval/{want} is not among the files checked, so a regression there would be "
            f"invisible")
    assert not bad, f"the live tree has {len(bad)} violation(s): {bad[:2]}"
    print(f"selftest OK: the real pre-fix path is rejected and the real fixed one accepted, "
          f"a prose mention is not a call site; {len(files)} live file(s) pass")
    return 0


if __name__ == "__main__":
    sys.exit(_selftest() if "--selftest" in sys.argv else main())
