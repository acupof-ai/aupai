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
    "facts/*.json",
    "scripts/*.json",
    "runs/*.jsonl",
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
    lines = [f"{sha_head(root, p)}  {p}" for p in scoped_paths(root)]
    with open(MANIFEST, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return len(lines)


def write_manifest_index(root=ROOT):
    """Manifest from the index (staged blobs), not HEAD. Used by the pre-commit
    hook so the committed manifest matches HEAD after the commit lands."""
    lines = []
    for p in scoped_paths(root):
        sha = sha_index(root, p)
        if sha:
            lines.append(f"{sha}  {p}")
    with open(MANIFEST, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return len(lines)


def read_manifest(path=None):
    if path is None:
        path = MANIFEST
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
    """Every file the manifest names must exist here and match. The pod runs this.
    Also reports .py files no manifest entry names -- the forward check cannot see
    them, and a bare-podput script otherwise runs unregistered indefinitely.

    runs/*.jsonl drift is expected (the pod produces rows; the committed copy lags).
    It is reported, not failed -- a permanent red is no signal. Code drift FAILs."""
    manifest_path = os.path.join(root, "data", "pod_head_manifest.txt")
    if not os.path.exists(manifest_path):
        return False, f"no manifest at {os.path.relpath(manifest_path, root)}"
    manifest = read_manifest(manifest_path)
    bad = []
    runs_div = []
    for p, want in manifest.items():
        fp = os.path.join(root, p)
        if not os.path.exists(fp):
            bad.append(f"missing {p}")
        elif sha_disk(fp) != want:
            if p.startswith("runs/"):
                runs_div.append(p)
            else:
                bad.append(f"diff {p}")
    if bad:
        return False, f"{len(bad)} drifted: {'; '.join(bad)}"
    extra = unregistered_py(root, manifest)
    parts = [f"{len(manifest)} files match"]
    if runs_div:
        parts.append(f"{len(runs_div)} runs file(s) diverged (pod produces rows; sync to commit): {'; '.join(sorted(runs_div))}")
    if extra:
        shown = ", ".join(extra[:5]) + ("..." if len(extra) > 5 else "")
        parts.append(f"{len(extra)} UNREGISTERED .py not in manifest: {shown}")
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


def selftest():
    """The reverse scan on a real-shaped world: one registered script, one unregistered
    probe at root, one .py in an excluded dir and one under data/ -- only the probe is
    unregistered, and check_pod still passes while naming it."""
    import tempfile

    d = tempfile.mkdtemp()
    for sub in ("scripts", "datagen", "data"):
        os.makedirs(os.path.join(d, sub), exist_ok=True)
    open(os.path.join(d, "scripts", "real.py"), "w").write("# registered\n")
    open(os.path.join(d, "probe.py"), "w").write("# throwaway\n")
    open(os.path.join(d, "datagen", "gen.py"), "w").write("# excluded dir\n")
    open(os.path.join(d, "data", "tool.py"), "w").write("# data dir\n")
    manifest = {"scripts/real.py": sha_disk(os.path.join(d, "scripts", "real.py"))}
    found = unregistered_py(d, manifest)
    assert found == ["probe.py"], found
    with open(os.path.join(d, "data", "pod_head_manifest.txt"), "w") as f:
        f.write("".join(f"{sha}  {p}\n" for p, sha in manifest.items()))
    ok, evidence = check_pod(d)
    assert ok and "UNREGISTERED" in evidence, evidence
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
