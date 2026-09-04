"""Population of check_selftests_are_gated, recomputed independently.

The check's own population is: tracked .py files whose docstring-stripped body contains
"--selftest", plus tracked test_*.py with a module-level `if __name__`. This recomputes
that set, and separately asks the question the check does NOT ask: does any tracked .sh
carry a runnable selftest, and does any file OUTSIDE .py carry one?

  python3 runs/audit_0904/selftest_population.py
  python3 runs/audit_0904/selftest_population.py --selftest
"""

import os
import re
import subprocess
import sys

ROOT = os.environ.get(
    "AUDIT_ROOT",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import harness as H  # noqa: E402


def tracked(root=None):
    root = root or ROOT
    r = subprocess.run(["git", "ls-files"], cwd=root, capture_output=True, text=True, check=True)
    return [ln for ln in r.stdout.splitlines() if ln.strip()]


def read(root, rel):
    try:
        return open(os.path.join(root, rel), encoding="utf-8", errors="replace").read()
    except OSError:
        return ""


def measure(root=None):
    root = root or ROOT
    files = tracked(root)
    py = [f for f in files if f.endswith(".py")]
    sh = [f for f in files if f.endswith(".sh")]
    other = [f for f in files if not f.endswith((".py", ".sh", ".md", ".json", ".jsonl", ".txt", ".log"))]

    flag_py, runnable_test_py = set(), set()
    for rel in py:
        body = read(root, rel)
        if "--selftest" in H.strip_docstrings(body):
            flag_py.add(rel)
        elif re.search(r"(^|/)test_[\w-]+\.py$", rel) and re.search(
            r"^if __name__", H.strip_docstrings(body), re.M
        ):
            runnable_test_py.add(rel)

    # The half the check cannot see: a shell script with its own selftest path.
    flag_sh = {rel for rel in sh if "--selftest" in read(root, rel) or "selftest" in read(root, rel)}
    flag_other = {rel for rel in other if "--selftest" in read(root, rel)}

    # And the hook's maps, parsed the same way the check parses them.
    hook = read(root, os.path.join("scripts", "hooks", "pre-commit"))
    m = re.search(r"SELFTEST_FILES\s*=\s*\{([^}]*)\}", hook)
    nd = re.search(r"NEEDS_DATA\s*=\s*\{(.*?)\n    \}", hook, re.S)
    gated = set(re.findall(r'"([^"]+)"', m.group(1))) if m else set()
    needs = set(re.findall(r'"([^"]+)":', nd.group(1))) if nd else set()

    return {
        "tracked": len(files),
        "py": len(py),
        "sh": len(sh),
        "flag_py": flag_py,
        "runnable_test_py": runnable_test_py,
        "flag_sh": flag_sh,
        "flag_other": flag_other,
        "gated": gated,
        "needs": needs,
    }


def _selftest():
    d = measure()
    assert d["py"] > 200, f"only {d['py']} tracked .py -- ls-files is not reading the repo"
    # Known answer, read by hand: scripts/harness.py carries --selftest and is in NEEDS_DATA.
    assert "scripts/harness.py" in d["flag_py"], "harness.py itself is not in the flag set"
    # Known answer: scripts/test_resume_accumulates.py deliberately carries NO flag, so it
    # must land in the runnable-test half and not the flag half.
    t = "scripts/test_resume_accumulates.py"
    if os.path.exists(os.path.join(ROOT, t)):
        assert t not in d["flag_py"], f"{t} counted as flag-carrying; its docstring only mentions it"
        assert t in d["runnable_test_py"], f"{t} missing from the runnable-test half"
    print("selftest_population selftest ok")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
        raise SystemExit(0)
    d = measure()
    print(f"tracked {d['tracked']}  .py {d['py']}  .sh {d['sh']}")
    print(f"flag-carrying .py            : {len(d['flag_py'])}")
    print(f"runnable test_*.py, no flag  : {len(d['runnable_test_py'])}")
    print(f"hook SELFTEST_FILES entries  : {len(d['gated'])}")
    print(f"hook NEEDS_DATA entries      : {len(d['needs'])}")
    pop = d["flag_py"] | d["runnable_test_py"]
    ungated = sorted(pop - d["gated"] - d["needs"])
    print(f"\nthe check's population       : {len(pop)}; ungated {len(ungated)}")
    for x in ungated:
        print(f"   UNGATED {x}")
    print(f"\n.sh mentioning selftest (OUTSIDE the check's population): {len(d['flag_sh'])}")
    for x in sorted(d["flag_sh"]):
        print(f"   {x}")
    print(f"non-.py/.sh/.md/.json carrying --selftest: {len(d['flag_other'])}")
    for x in sorted(d["flag_other"]):
        print(f"   {x}")
