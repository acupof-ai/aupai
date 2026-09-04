#!/usr/bin/env python3
# restartable: one small JSON claim file per write; an interrupt mid-write leaves it malformed
# and claim() reads a malformed file as absent, so restart is cheap. Nothing to shard.
"""File claims: who may edit a listed shared file (AGENTS.md "Shared files", T0, 2026-09-04).

WHY. The rule "announce before editing train.py/sft*.py/AGENTS.md" was prose: an announcement
lived in a conversation, and three recorded incidents show what happens when it is a message
rather than a file -- a push rolled back 3b's datagen/build_corpus.py feature (2026-08-30),
`git checkout` erased another session's uncommitted device gate (2026-08-31), and `git add -A`
swept five sessions' uncommitted work into one commit (d535674). A claim makes the announce a
FILE on the shared tree runs/claims/files/, so a commit that touches a listed file without a
live claim by the committer refuses at pre-commit: coordination stops being remembered and
becomes checked.

A claim is per (tree, path): it lives in runs/claims/ of the tree the commit runs from.
acquire writes the claim; release removes it; a claim older than FILE_CLAIM_TTL is stale and
does not satisfy the hook. A session hands the file back (release) when it merges.

    python3 scripts/file_claim.py acquire --path train.py [--owner <name>]
    python3 scripts/file_claim.py release --path train.py
    python3 scripts/file_claim.py release-all --owner <name>   # merge_main.sh: hand back
    python3 scripts/file_claim.py status
    python3 scripts/file_claim.py --selftest      # acquire/refuse/release/stale, no real files
"""

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLAIM_DIR = os.path.join(ROOT, "runs", "claims", "files")
FILE_CLAIM_TTL = 6 * 3600  # a claim 6h old no longer satisfies the hook


def _claim_dir():
    # FILE_CLAIM_DIR lets the selftest run against a temp tree without touching real claims.
    return os.environ.get("FILE_CLAIM_DIR") or CLAIM_DIR


#: The files the "Shared files" rule guards (AGENTS.md). Announce = claim; a listed path
#: touched by a commit with no live claim refuses at pre-commit.
SHARED_FILES = ("train.py", "sft.py", "sft_math.py", "AGENTS.md")


def _owner():
    # $USER is `bytedance` for every session on this box, so scoping release-all by it is no
    # scoping at all: one session's merge hands back every other session's claims, and with the
    # 6h TTL a leaked claim blocks all sessions' shared-file commits. Same identity model as
    # launch_gate._launch_owner: LAUNCH_OWNER wins, else the worktree's own name (one session
    # per worktree is the standing rule), `aupai-3b` -> `3b`.
    o = os.environ.get("LAUNCH_OWNER", "").strip()
    if o:
        return o
    base = os.path.basename(ROOT)
    return base[len("aupai-"):] if base.startswith("aupai-") else base


def claim_path(path):
    safe = path.replace("/", "__")
    return os.path.join(_claim_dir(), safe + ".json")


def acquire(path, owner=None):
    os.makedirs(_claim_dir(), exist_ok=True)
    cp = claim_path(path)
    owner = owner or _owner()
    live = claim(path)
    if live:
        return False, f"{path} already claimed by {live.get('owner')} at {live.get('time')}"
    rec = {"path": path, "owner": owner, "time": int(time.time())}
    with open(cp, "w", encoding="utf-8") as fh:
        json.dump(rec, fh)
    return True, f"claimed {path} for {owner}"


def release(path):
    cp = claim_path(path)
    if not os.path.exists(cp):
        return False, f"no claim for {path}"
    os.remove(cp)
    return True, f"released {path}"


def claim(path):
    """Live claim for path, or None. A stale claim reads as no claim (the hook ignores it)."""
    cp = claim_path(path)
    if not os.path.exists(cp):
        return None
    try:
        rec = json.load(open(cp, encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if time.time() - rec.get("time", 0) > FILE_CLAIM_TTL:
        return None
    return rec


def claims():
    d = _claim_dir()
    out = {}
    if not os.path.isdir(d):
        return out
    for name in sorted(os.listdir(d)):
        if not name.endswith(".json"):
            continue
        try:
            rec = json.load(open(os.path.join(d, name), encoding="utf-8"))
        except (OSError, ValueError):
            continue
        out[rec.get("path")] = rec
    return out


def release_all(owner, paths=None):
    """Release every claim owned by `owner` (all guarded paths, or the listed ones).

    merge_main.sh calls this after a merge to hand shared files back: the claim that
    let the branch commit pass is no longer needed once the edit is merged.
    """
    if paths is None:
        paths = SHARED_FILES
    removed = []
    for p in paths:
        rec = claim(p)
        if rec and rec.get("owner") == owner:
            release(p)
            removed.append(p)
    return removed


def main(argv=None):
    ap = argparse.ArgumentParser(prog="file_claim")
    # The pre-commit hook's SELFTEST_FILES invoker passes `--selftest` (the repo-standard
    # flag); keep a positional `selftest` too so a bare `file_claim.py selftest` works by hand.
    ap.add_argument("cmd", choices=["acquire", "release", "release-all", "status", "selftest"],
                    nargs="?")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--path", help="the shared file (relative to the repo root)")
    ap.add_argument("--owner", default=None)
    a, _ = ap.parse_known_args(argv)
    if a.selftest or a.cmd == "selftest":
        return _selftest()
    if a.cmd == "acquire":
        assert a.path, "acquire needs --path"
        ok, msg = acquire(a.path, a.owner)
        print(msg)
        return 0 if ok else 1
    if a.cmd == "release":
        assert a.path, "release needs --path"
        ok, msg = release(a.path)
        print(msg)
        return 0 if ok else 1
    if a.cmd == "release-all":
        removed = release_all(a.owner or _owner())
        print(f"released {len(removed)} claim(s): {', '.join(removed) if removed else 'none'}")
        return 0
    for p, rec in claims().items():
        print(f"{p:16s} owner={rec.get('owner')} t={time.strftime('%H:%M', time.gmtime(rec.get('time')))} UTC")
    return 0


def _selftest():
    import tempfile
    d = tempfile.mkdtemp(prefix="file_claim_st_")
    import subprocess
    fails = []
    def run(*args):
        r = subprocess.run([sys.executable, os.path.abspath(__file__)] + list(args),
                           capture_output=True, text=True)
        return r.returncode, (r.stdout + r.stderr)
    try:
        # Hermetic: point the claim dir at the temp tree (the re-exec children inherit the env),
        # and the owner at a fixed name, so the selftest never reads or writes real claims.
        # The owner is pinned via LAUNCH_OWNER, not $USER: $USER is no longer the default owner,
        # and this assertion used to read `st_owner` straight out of it -- the world encoded the
        # very defect (unscoped release-all) that the last two worlds below now catch.
        os.environ["FILE_CLAIM_DIR"] = d
        os.environ["USER"] = "st_owner"
        os.environ["LAUNCH_OWNER"] = "st_owner"
        # acquire a path -> claim present, second acquire refuses
        rc, out = run("acquire", "--path", "train.py")
        if rc != 0: fails.append(f"acquire failed: {out}")
        c = claim("train.py")
        if not (c and c["owner"] == "st_owner"): fails.append("claim not recorded with owner")
        rc, _ = run("acquire", "--path", "train.py")
        if rc == 0: fails.append("double acquire must refuse")
        # release -> gone
        rc, _ = run("release", "--path", "train.py")
        if rc != 0: fails.append("release failed")
        if claim("train.py"): fails.append("claim survived release")
        # a stale claim reads as no claim (TTL)
        os.makedirs(os.path.dirname(claim_path("sft.py")), exist_ok=True)
        with open(claim_path("sft.py"), "w") as fh:
            json.dump({"path": "sft.py", "owner": "old", "time": int(time.time()) - FILE_CLAIM_TTL - 10}, fh)
        if claim("sft.py"): fails.append("stale claim must read as absent")
        # an unlisted path still records but the hook only guards SHARED_FILES
        # release-all hands back only OWNER claims, leaves others
        rc, _ = run("acquire", "--path", "AGENTS.md", "--owner", "st_owner")
        if rc != 0: fails.append("acquire AGENTS.md failed")
        rc, _ = run("acquire", "--path", "train.py", "--owner", "other")
        if rc != 0: fails.append("acquire train.py(other) failed")
        rc, out = run("release-all", "--owner", "st_owner")
        if rc != 0: fails.append(f"release-all failed: {out}")
        if claim("AGENTS.md"): fails.append("release-all left st_owner claim")
        if not claim("train.py"): fails.append("release-all dropped a foreign claim")
        rc, _ = run("release", "--path", "train.py")
        if rc != 0: fails.append("release train.py(other) failed")
        # THE DEFAULT OWNER IGNORES $USER. The worlds above all pass --owner explicitly, so they
        # pass just as well when the default is $USER -- which is `bytedance` for every session
        # here, i.e. no scoping at all. LAUNCH_OWNER wins; with it unset the worktree name is
        # used; neither may be the $USER value.
        os.environ["LAUNCH_OWNER"] = "st_wt"
        rc, _ = run("acquire", "--path", "sft_math.py")
        if rc != 0: fails.append("acquire under LAUNCH_OWNER failed")
        c = claim("sft_math.py")
        if not (c and c["owner"] == "st_wt"):
            fails.append(f"LAUNCH_OWNER must win, got {c and c.get('owner')}")
        os.environ.pop("LAUNCH_OWNER")
        rc, _ = run("acquire", "--path", "train.py")
        if rc != 0: fails.append("acquire under default owner failed")
        c = claim("train.py")
        if c and c["owner"] == "st_owner":
            fails.append("default owner is $USER -- release-all would not be scoped")
        rc, out = run("release-all", "--owner", "st_wt")
        if rc != 0: fails.append(f"release-all(st_wt) failed: {out}")
        if claim("sft_math.py"): fails.append("release-all left the LAUNCH_OWNER claim")
        if not claim("train.py"): fails.append("release-all dropped a foreign default-owner claim")
    finally:
        for _k in ("FILE_CLAIM_DIR", "USER", "LAUNCH_OWNER"):
            try:
                os.environ.pop(_k)
            except KeyError:
                pass
        import shutil; shutil.rmtree(d, ignore_errors=True)
    for f in fails:
        print(f"BUG {f}", file=sys.stderr)
    print(f"file_claim selftest: {'PASS (8 worlds)' if not fails else f'{len(fails)} BUG(S)'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())