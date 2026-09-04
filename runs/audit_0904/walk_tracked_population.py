"""walk_tracked's population vs the name "tracked", and vs each caller's rule.

harness.py:1726 `walk_tracked(root, suffixes)` is the shared population helper for
curl_ipv4, timestamps_are_utc, no_conflict_markers, selftests_are_gated and others. It is
an os.walk with a directory blacklist (harness.py:1723 `_SKIP_DIRS = {".git", "data",
"runs", ...}`) and it never consults git, so:

  1. it yields UNTRACKED files -- a scratch .py in the tree is judged as repo content
  2. it never yields anything under runs/ or data/ -- so a check that passes `.jsonl`
     or `.txt` in its suffix tuple is asking for a file class that lives only in the
     directories the walker refuses to enter

This measures both gaps against `git ls-files`, per suffix, so a caller's suffix tuple can
be read against what the walker can actually reach.

  python3 runs/audit_0904/walk_tracked_population.py
  python3 runs/audit_0904/walk_tracked_population.py --selftest
"""

import os
import subprocess
import sys

ROOT = os.environ.get(
    "AUDIT_ROOT",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import harness as H  # noqa: E402

SUFFIXES = (".md", ".py", ".json", ".jsonl", ".sh", ".txt", ".yml", ".yaml")


def git_tracked(root=None, suffixes=SUFFIXES):
    root = root or ROOT
    r = subprocess.run(["git", "ls-files"], cwd=root, capture_output=True, text=True, check=True)
    return {ln for ln in r.stdout.splitlines() if ln.strip() and ln.endswith(suffixes)}


def walked(root=None, suffixes=SUFFIXES):
    root = root or ROOT
    return {os.path.relpath(p, root) for p, _ in H.walk_tracked(root, suffixes)}


def measure(root=None):
    root = root or ROOT
    g, w = git_tracked(root), walked(root)
    per = {}
    for s in SUFFIXES:
        gs = {x for x in g if x.endswith(s)}
        ws = {x for x in w if x.endswith(s)}
        per[s] = {
            "tracked": len(gs),
            "walked": len(ws),
            "tracked_not_walked": sorted(gs - ws),
            "walked_not_tracked": sorted(ws - gs),
        }
    return {"git": g, "walk": w, "per_suffix": per}


def _selftest():
    d = measure()
    g, w = d["git"], d["walk"]
    assert len(g) > 400, f"git ls-files returned {len(g)} matching files -- not reading the repo"
    assert len(w) > 200, f"walk_tracked yielded {len(w)} -- not reading the tree"
    # KNOWN ANSWER, read from harness.py:1723 before running this: `runs` is in _SKIP_DIRS,
    # so a tracked file under runs/ must be in tracked-not-walked. If it is not, this
    # instrument is not calling the real helper.
    runs_tracked = {x for x in g if x.startswith("runs/")}
    assert runs_tracked, "no tracked file under runs/ -- the fixture assumption is wrong"
    assert not (runs_tracked & w), (
        f"walk_tracked yielded {len(runs_tracked & w)} file(s) under runs/, but harness.py:1723 "
        f"lists `runs` in _SKIP_DIRS -- this scan is not measuring the real helper"
    )
    # And the negative half: a tracked .py at the repo root MUST be walked, or the walker
    # is broken rather than merely narrow.
    root_py = {x for x in g if x.endswith(".py") and "/" not in x}
    assert root_py and root_py <= w, f"walk_tracked missed root .py files: {sorted(root_py - w)}"
    print(f"walk_tracked_population selftest ok ({len(g)} tracked, {len(w)} walked)")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
        raise SystemExit(0)
    d = measure()
    print(f"_SKIP_DIRS = {sorted(H._SKIP_DIRS)}")
    print(f"git ls-files matching {SUFFIXES}: {len(d['git'])}")
    print(f"walk_tracked yields              : {len(d['walk'])}\n")
    print(f"{'suffix':8s} {'tracked':>8s} {'walked':>7s} {'tracked-not-walked':>19s} {'walked-not-tracked':>19s}")
    for s, v in d["per_suffix"].items():
        print(f"{s:8s} {v['tracked']:8d} {v['walked']:7d} {len(v['tracked_not_walked']):19d} "
              f"{len(v['walked_not_tracked']):19d}")
    print("\nUNTRACKED files the walker judges as repo content:")
    unt = sorted(d["walk"] - d["git"])
    for x in unt[:25]:
        print(f"   {x}")
    if len(unt) > 25:
        print(f"   ... {len(unt) - 25} more")
    print("\nTracked files the walker cannot reach, by top directory:")
    top = {}
    for x in sorted(d["git"] - d["walk"]):
        top.setdefault(x.split("/")[0], []).append(x)
    for k, v in sorted(top.items(), key=lambda kv: -len(kv[1])):
        print(f"   {k:22s} {len(v)}")
