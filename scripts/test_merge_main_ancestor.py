#!/usr/bin/env python3
"""merge_main.sh warns when the named branch is already an ancestor of main.

    python3 scripts/test_merge_main_ancestor.py [--selftest]

WHY THIS EXISTS. 2026-09-04: `scripts/merge_main.sh b0` merged b0 at ccbc0891, already in main,
while b0's real work sat on b0-ve-rownorms. git printed "Already up to date.", the script exited
0, and the mistake surfaced minutes later through pod_push's unrelated "differs from main"
refusal. Nothing shipped and nothing said so.

WHY IT DRIVES A REAL REPOSITORY. The property is about git ancestry and about the ORDER of two
git calls, and the ordering is the whole content of the fix: after a fast-forward merge the branch
IS an ancestor, so the same test run afterwards would be true of every successful merge. A fixture
that stubs git cannot show that; two real branches can.

THE NEGATIVE CONTROL, named per tilerl's rule (2026-09-04): case 2 below is a branch that is NOT
an ancestor, and it must produce NO warning. Delete the `merge-base --is-ancestor` guard and case
1 fails; make the guard unconditional and case 2 fails. Either mutation is caught.
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCRIPT = os.path.join(HERE, "merge_main.sh")
ENV = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}


def git(d, *a, check=True):
    r = subprocess.run(["git", "-C", d, *a], capture_output=True, text=True, env=ENV)
    if check and r.returncode:
        raise AssertionError(f"git {' '.join(a)} failed in {d}: {r.stderr[:200]}")
    return r


def _repo():
    """A repo with main plus two branches: `merged` (already in main) and `ahead` (not)."""
    d = tempfile.mkdtemp(prefix="mergemain_")
    git(d, "init", "-q", "-b", "main")
    git(d, "config", "user.email", "t@t")
    git(d, "config", "user.name", "t")
    with open(os.path.join(d, "a.txt"), "w") as f:
        f.write("1\n")
    git(d, "add", "a.txt")
    git(d, "commit", "-q", "-m", "base")

    # `merged`: a commit that main then absorbs, so it becomes an ancestor.
    git(d, "checkout", "-q", "-b", "merged")
    with open(os.path.join(d, "b.txt"), "w") as f:
        f.write("1\n")
    git(d, "add", "b.txt")
    git(d, "commit", "-q", "-m", "the work that already landed")
    git(d, "checkout", "-q", "main")
    git(d, "merge", "-q", "--no-edit", "merged")

    # `ahead`: a commit main does not have.
    git(d, "checkout", "-q", "-b", "ahead")
    with open(os.path.join(d, "c.txt"), "w") as f:
        f.write("1\n")
    git(d, "add", "c.txt")
    git(d, "commit", "-q", "-m", "work main has not seen")
    git(d, "checkout", "-q", "main")
    return d


def _run(repo, branch):
    """merge_main.sh against `repo` instead of the real integration tree.

    MAIN is a literal in the script, so the copy rewrites exactly that line. Asserted, not
    assumed: if the assignment's shape changes the substitution silently does nothing and this
    test would then exercise the REAL integration tree, which is the worst possible failure for
    a test to have.
    """
    src = open(SCRIPT, encoding="utf-8").read()
    assert re.search(r"^MAIN=/Users/bytedance/code/aupai$", src, re.M), (
        "merge_main.sh's MAIN assignment is not the line this test rewrites; fix the "
        "substitution before trusting a green run -- otherwise it merges into the real tree")
    src = re.sub(r"^MAIN=.*$", f"MAIN={repo}", src, count=1, flags=re.M)
    p = os.path.join(repo, "merge_main_under_test.sh")
    with open(p, "w", encoding="utf-8") as f:
        f.write(src)
    os.chmod(p, 0o755)
    r = subprocess.run(["bash", p, branch], capture_output=True, text=True, env=ENV, cwd=repo)
    return r.returncode, r.stdout + r.stderr


def main():
    d = _repo()
    try:
        # 1. An ancestor: warned, named, and still exit 0.
        rc, out = _run(d, "merged")
        tip = git(d, "rev-parse", "--short", "merged").stdout.strip()
        assert "already an ancestor" in out, (
            f"merging an ancestor printed no warning -- this is the b0/ccbc0891 case, where "
            f"'Already up to date.' was the only output:\n{out}")
        assert tip in out, f"the warning does not name the tip {tip}:\n{out}"
        assert "the work that already landed" in out, (
            f"the warning does not show the tip's subject, which is what tells the operator "
            f"they named the wrong branch:\n{out}")
        assert rc == 0, f"the warning must not refuse: rc={rc}\n{out}"

        # 2. NEGATIVE CONTROL: a branch main does not have must NOT be warned about. This is
        #    the case an unconditional warning breaks, and the case an ordering mistake breaks
        #    -- put the check after the merge and this branch becomes an ancestor by then.
        rc, out = _run(d, "ahead")
        assert "already an ancestor" not in out, (
            f"a branch main does not have was warned about; a warning on every merge is noise "
            f"and would be ignored within a day:\n{out}")
        assert rc == 0, f"a real merge failed: rc={rc}\n{out}"
        r = subprocess.run(["git", "-C", d, "merge-base", "--is-ancestor", "ahead", "main"],
                           capture_output=True, env=ENV)
        assert r.returncode == 0, "the second case did not actually merge, so it proves nothing"

        # 3. The ORDER, asserted directly on the source: the guard must precede the merge.
        src = open(SCRIPT, encoding="utf-8").read()
        i_guard = src.index("--is-ancestor")
        i_merge = src.index('merge --no-edit "$1"')
        assert i_guard < i_merge, (
            "the ancestor check is written after the merge, where a fast-forward has already "
            "made every merged branch an ancestor -- it would fire on every successful merge")
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print("merge_main ancestor: an already-merged branch warns and names its tip and subject "
          "at exit 0; a branch main lacks is not warned; the guard precedes the merge")


if __name__ == "__main__":
    if sys.argv[1:] not in ([], ["--selftest"]):
        sys.exit(f"usage: {os.path.basename(__file__)} [--selftest]  (got {sys.argv[1:]})")
    main()
