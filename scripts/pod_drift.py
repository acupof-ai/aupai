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

A dev checkout (has .git, no CI) skips both: uncommitted changes are normal there.

Usage:
  python scripts/pod_drift.py --write        # regenerate the manifest from HEAD
  python scripts/pod_drift.py --check        # pod gate: files here vs the manifest
  python scripts/pod_drift.py --check-head   # CI gate: manifest vs HEAD
  python scripts/pod_drift.py --list-scoped  # the file set, one per line
"""
import hashlib
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "data", "pod_head_manifest.txt")

# Files that execute on the pod (pretrain -> score flow). Mirrors the old
# pod_sync_check.sh scope; datagen/filters/mathbank/workflows never run there.
SCOPE = [
    "*.py",
    "*.sh",
    "data/mix_scale_*.json",
    "data/tokenizer.json",
    "AGENTS.md",
    "docs/standards/*.md",
    ":!scripts/pod_sync_check.sh",
]
EXCLUDE_DIRS = ("datagen", "filters", "mathbank", "workflows")


def scoped_paths(root=ROOT):
    out = subprocess.run(
        ["git", "ls-files", *SCOPE], cwd=root, capture_output=True, text=True, check=True
    ).stdout.split()
    return sorted(p for p in out if not p.startswith(EXCLUDE_DIRS))


def _sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha_head(root, path):
    b = subprocess.run(
        ["git", "show", f"HEAD:{path}"], cwd=root, capture_output=True, check=True
    ).stdout
    return _sha_bytes(b)


def sha_disk(path):
    with open(path, "rb") as f:
        return _sha_bytes(f.read())


def write_manifest(root=ROOT):
    lines = [f"{sha_head(root, p)}  {p}" for p in scoped_paths(root)]
    with open(MANIFEST, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return len(lines)


def read_manifest(path=MANIFEST):
    m = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            sha, p = line.split("  ", 1)
            m[p] = sha
    return m


def check_pod(root=ROOT):
    """Every file the manifest names must exist here and match. The pod runs this."""
    manifest_path = os.path.join(root, "data", "pod_head_manifest.txt")
    if not os.path.exists(manifest_path):
        return False, f"no manifest at {os.path.relpath(manifest_path, root)}"
    bad = []
    for p, want in read_manifest(manifest_path).items():
        fp = os.path.join(root, p)
        if not os.path.exists(fp):
            bad.append(f"missing {p}")
        elif sha_disk(fp) != want:
            bad.append(f"diff {p}")
    if bad:
        return False, f"{len(bad)} drifted (first: {bad[0]})"
    return True, f"{len(read_manifest(manifest_path))} files match the manifest"


def check_head(root=ROOT):
    """The committed manifest must describe HEAD. CI runs this."""
    if not os.path.exists(MANIFEST):
        return False, "no manifest; run scripts/pod_drift.py --write"
    have = read_manifest()
    want = {p: sha_head(root, p) for p in scoped_paths(root)}
    stale = [p for p in want if have.get(p) != want[p]]
    gone = [p for p in have if p not in want]
    if stale or gone:
        return False, f"manifest stale: {len(stale)} changed, {len(gone)} removed; run --write"
    return True, f"manifest matches HEAD ({len(want)} files)"


def is_pod(root=ROOT):
    """A git checkout has .git; the pod's hand-pushed tree does not."""
    return not os.path.isdir(os.path.join(root, ".git"))


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "--check"
    if mode == "--write":
        n = write_manifest()
        print(f"wrote {n} entries to {os.path.relpath(MANIFEST, ROOT)}")
    elif mode == "--list-scoped":
        print("\n".join(scoped_paths()))
    elif mode == "--check":
        if is_pod(ROOT):
            ok, evidence = check_pod(ROOT)
        elif os.environ.get("CI") == "true":
            ok, evidence = check_head(ROOT)
        else:
            print("dev checkout: nothing to check (CI gates manifest freshness, the pod gates file drift)")
            sys.exit(0)
        print(("OK: " if ok else "DRIFT: ") + evidence)
        sys.exit(0 if ok else 1)
    elif mode == "--check-head":
        ok, evidence = check_head(ROOT)
        print(("OK: " if ok else "STALE: ") + evidence)
        sys.exit(0 if ok else 1)
    else:
        print(__doc__)
        sys.exit(2)


if __name__ == "__main__":
    main()
