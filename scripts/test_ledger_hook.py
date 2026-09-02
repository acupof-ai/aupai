#!/usr/bin/env python3
"""The five de-33 acceptance readings, pre-registered by 1e. Invokes the hooks AS PROGRAMS.

    python3 scripts/test_ledger_hook.py --selftest

# restartable: builds temp git repos, reads this repo's history. Costs ~10s.

WHY NOT A GREEN COMMIT AS EVIDENCE. `.git/hooks/pre-commit` is a symlink resolved against MAIN's
worktree, so a hook edited on a branch does not run -- and a commit that prints five green hook
lines proves nothing about the version in the branch. Both hooks are therefore executed directly,
in temp repos built from this repo's real blobs.

The five readings:

  1  a59ac1f's shape staged        pre-commit rc!=0, names (file, key)
  2  c3a5a23's shape staged        pre-commit rc=0
  3  loss + AUPAI_LEDGER_REWRITE   pre-commit rc=0, prints the list, drops the marker file
  4  same, message lacks marker    commit-msg rc!=0
  5  a duplicate key ADDED         pre-commit rc!=0, names the key

Reading 5 judges the DELTA, not the presence: score_matrix.jsonl already carries 22 duplicates
on main, so "any duplicate refuses" would refuse every commit touching the file, including the
folding commit that removes them.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRE = os.path.join(ROOT, "scripts", "hooks", "pre-commit")
MSG_HOOK = os.path.join(ROOT, "scripts", "hooks", "commit-msg")
EXP = "runs/experiments.jsonl"
SM = "runs/score_matrix.jsonl"
ENV = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
ENV.update(GIT_CONFIG_GLOBAL="/dev/null", GIT_CONFIG_NOSYSTEM="1")


def blob(rev, path):
    r = subprocess.run(["git", "show", f"{rev}:{path}"], cwd=ROOT, capture_output=True, text=True)
    return None if r.returncode else r.stdout


def world(files, head_files=None):
    """A repo whose HEAD holds head_files (default: files) and whose index holds files.

    scripts/ and .gitattributes are copied in because the hook imports ledger_audit from the
    repo it is running in -- a fixture without them would exercise the ImportError branch and
    report SKIP, which reads as a pass.
    """
    d = tempfile.mkdtemp(prefix="ledgerhook_")
    os.makedirs(os.path.join(d, "runs"), exist_ok=True)
    os.makedirs(os.path.join(d, "scripts", "hooks"), exist_ok=True)
    for n in ("ledger_audit.py", "test_ledger_predicates.py"):
        shutil.copy(os.path.join(ROOT, "scripts", n), os.path.join(d, "scripts", n))
    shutil.copy(os.path.join(ROOT, ".gitattributes"), os.path.join(d, ".gitattributes"))
    for rel, text in (head_files or files).items():
        with open(os.path.join(d, rel), "w", encoding="utf-8") as fh:
            fh.write(text)
    subprocess.run(["git", "init", "-q"], cwd=d, env=ENV)
    subprocess.run(["git", "add", "-A"], cwd=d, env=ENV)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m",
                    "base"], cwd=d, env=ENV)
    if head_files:
        for rel, text in files.items():
            with open(os.path.join(d, rel), "w", encoding="utf-8") as fh:
                fh.write(text)
        subprocess.run(["git", "add", *files], cwd=d, env=ENV)
    return d


def run(hook, d, args=(), env=None):
    e = dict(ENV)
    if env:
        e.update(env)
    r = subprocess.run([sys.executable, hook, *args], cwd=d, capture_output=True, text=True, env=e)
    return r.returncode, r.stdout + r.stderr


def marker_file(d):
    p = subprocess.run(["git", "rev-parse", "--git-path", "AUPAI_LEDGER_REGRESSED"],
                       cwd=d, capture_output=True, text=True, env=ENV).stdout.strip()
    return os.path.join(d, p) if p and not os.path.isabs(p) else p


def _selftest():
    fails = []
    before, after = blob("a59ac1f^", EXP), blob("a59ac1f", EXP)
    good_b, good_a = blob("c3a5a23^", EXP), blob("c3a5a23", EXP)
    if not all((before, after, good_b, good_a)):
        print("SKIP: a required revision is absent from this clone")
        return 0

    # 1. The incident's shape refuses and names the key.
    d = world({EXP: after}, head_files={EXP: before})
    rc, out = run(PRE, d)
    ok = rc != 0 and "REFUSING" in out and EXP in out and "ab_zeroinit" in out
    fails += [] if ok else [f"1: incident shape must refuse and name the key (rc={rc})"]
    print(f"  {'ok  ' if ok else 'BUG '} 1 a59ac1f shape: rc={rc}, names file+key="
          f"{EXP in out and 'ab_zeroinit' in out}")

    # 2. An ordinary append passes.
    d = world({EXP: good_a}, head_files={EXP: good_b})
    rc, out = run(PRE, d)
    ok = rc == 0 or "REFUSING" not in out
    fails += [] if ok else [f"2: an appended done event must pass (rc={rc}, {out[:200]})"]
    print(f"  {'ok  ' if ok else 'BUG '} 2 c3a5a23 shape: rc={rc}, no ledger refusal="
          f"{'REFUSING' not in out}")

    # 3. The env var allows the loss, prints the list, and drops the marker.
    d = world({EXP: after}, head_files={EXP: before})
    rc, out = run(PRE, d, env={"AUPAI_LEDGER_REWRITE": "acceptance reading 3"})
    mk = marker_file(d)
    dropped = bool(mk) and os.path.exists(mk)
    ok = "ALLOWING" in out and "ab_zeroinit" in out and dropped
    fails += [] if ok else [f"3: env var must allow, list, and mark (rc={rc}, marker={dropped})"]
    print(f"  {'ok  ' if ok else 'BUG '} 3 env var allows: listed={'ab_zeroinit' in out}, "
          f"marker dropped={dropped}")

    # 4. commit-msg refuses when the message does not carry the reason (marker still present).
    if dropped:
        mf = os.path.join(d, "MSG")
        with open(mf, "w", encoding="utf-8") as fh:
            fh.write("sync: fold rows\n")
        rc, out = run(MSG_HOOK, d, args=[mf])
        gone = not os.path.exists(mk)
        ok = rc != 0 and "REFUSING" in out and gone
        fails += [] if ok else [f"4: no marker in message must refuse (rc={rc}, consumed={gone})"]
        print(f"  {'ok  ' if ok else 'BUG '} 4 message lacks `ledger-rewrite:`: rc={rc}, "
              f"marker consumed={gone}")
    else:
        fails.append("4: skipped, reading 3 dropped no marker")

    # 5. A duplicate key ADDED to score_matrix refuses; a pre-existing one does not.
    sm = blob("main", SM)
    if sm:
        rows = [ln for ln in sm.splitlines() if ln.strip()]
        d = world({SM: "\n".join(rows + [rows[0]]) + "\n"}, head_files={SM: "\n".join(rows) + "\n"})
        rc, out = run(PRE, d)
        # Match on the KEY, not on the sentence: the wording changed once already (the count
        # phrasing went away when duplicates() returned bare keys), and a test pinned to prose
        # reports BUG for a guard that worked.
        ok = rc != 0 and "REFUSING" in out and "ckpt_0830v1_0.2b.pt" in out
        fails += [] if ok else [f"5: an ADDED duplicate must refuse (rc={rc}, {out[:200]})"]
        print(f"  {'ok  ' if ok else 'BUG '} 5 duplicate added: rc={rc}, "
              f"named={'ckpt_0830v1_0.2b.pt' in out}")

        d = world({SM: "\n".join(rows) + "\n"})
        with open(os.path.join(d, SM), "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"ckpt": "zz_new.pt", "profile": "full", "metrics": {}}) + "\n")
        subprocess.run(["git", "add", SM], cwd=d, env=ENV)
        rc, out = run(PRE, d)
        ok = rc == 0 or "more than one row under" not in out
        fails += [] if ok else [f"5b: pre-existing duplicates must not refuse (rc={rc})"]
        print(f"  {'ok  ' if ok else 'BUG '} 5b pre-existing duplicates pass: rc={rc}")
    else:
        fails.append("5: score_matrix.jsonl absent on main")

    for f in fails:
        print(f"BUG {f}", file=sys.stderr)
    print(f"\nde-33 hook acceptance: {'PASS (6 readings)' if not fails else f'{len(fails)} BUG(S)'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(_selftest())
