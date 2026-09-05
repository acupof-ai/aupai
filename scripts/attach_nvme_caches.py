#!/usr/bin/env python3
"""Re-attach the NVMe token-cache mount inside the container, and VERIFY IT BY READING.

Run from the HOST (tn exec), not the container: the mount is cloned in the host's namespace, where
/data02 is visible, and attached in the container's, where it is not. scripts/host_mount_into_container.py
does the syscalls; this adds idempotence and the verification, and is what bootstrap_pod.sh calls.

WHY A STAGE AND NOT A ONE-OFF. The mount lives exactly as long as the container. A restart drops it
silently -- the mount POINT survives as an ordinary empty directory on the overlay, so nothing
errors: train.py finds no cache at the NVMe path, decides to REBUILD 247.8 GB onto a disk that is
87% full, and the first symptom is the disk filling. That is why the check below reads rather than
looks.

VERIFY BY READING, TWO SIGNALS, both required (4c's ruling 2026-09-05):

  1. st_dev of the mount point differs from st_dev of "/". A mount that vanished leaves a directory
     whose st_dev IS the root's, so this separates "mounted" from "empty directory where a mount
     used to be" -- and `os.path.ismount` does not: it compares against the PARENT, and /mnt is
     itself on the overlay, so an empty /mnt/data02 and a mounted one both read as... whatever the
     parent says. Tested, not assumed.

  2. A known cache's sha256 prefix matches what the copy recorded. Signal 1 proves something is
     mounted; only this proves it is the filesystem holding the caches we verified. A different
     NVMe mounted at the same path passes signal 1 and fails this.

Signal 2 reads 8 MB, not a whole 85 GB cache: the prefix is what the copy record stores, so the
comparison is against a recorded quantity rather than a recomputed one.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PREFIX_BYTES = 8 << 20


def _sha_prefix(path, n=PREFIX_BYTES):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(n))
    return h.hexdigest()


def container_pid(cid_hint=None):
    """The container's host-side pid, from crictl. Not cached anywhere: it changes on every restart,
    which is exactly the event this script exists to recover from."""
    r = subprocess.run(["crictl", "ps", "-o", "json"], capture_output=True, text=True)
    if r.returncode != 0:
        return None, f"crictl ps failed: {r.stderr.strip()[:120]}"
    try:
        obj = json.loads(r.stdout)
    except ValueError as e:
        return None, f"crictl ps output is not json: {e}"
    cands = []
    for c in obj.get("containers", []):
        cid = c.get("id", "")
        name = (c.get("metadata") or {}).get("name", "")
        if cid_hint and not cid.startswith(cid_hint):
            continue
        cands.append((cid, name))
    if not cands:
        return None, "no running container matched"
    if len(cands) > 1 and not cid_hint:
        return None, ("several containers are running and none was named -- pass --container <id "
                      "prefix>:\n" + "\n".join(f"    {c[:16]}  {n}" for c, n in cands))
    cid = cands[0][0]
    r = subprocess.run(["crictl", "inspect", cid], capture_output=True, text=True)
    if r.returncode != 0:
        return None, f"crictl inspect failed: {r.stderr.strip()[:120]}"
    try:
        return json.loads(r.stdout)["info"]["pid"], None
    except (ValueError, KeyError) as e:
        return None, f"no pid in crictl inspect output: {e}"


def in_container(pid, script):
    """Run a python snippet inside the container's mount namespace via nsenter."""
    r = subprocess.run(["nsenter", "-t", str(pid), "-m", "python3", "-c", script],
                       capture_output=True, text=True)
    return r.returncode, r.stdout.strip(), r.stderr.strip()


VERIFY = r"""
import hashlib, json, os, sys
target, record = TARGET, RECORD
if not os.path.isdir(target):
    print("MISSING " + target); sys.exit(2)
# SIGNAL 1: st_dev, not os.path.ismount. ismount compares against the PARENT, and /mnt is on the
# overlay too, so it cannot separate a live mount from the empty directory a dropped mount leaves.
if os.stat(target).st_dev == os.stat("/").st_dev:
    print("NOT_MOUNTED " + target + " is on the root filesystem"); sys.exit(2)
# SIGNAL 2: a full sha256 the copy actually recorded. The record holds whole-file digests, so the
# comparison is against a recorded quantity -- which is the point: signal 1 proves SOMETHING is
# mounted, and only this proves it is the filesystem holding the caches that were verified. A
# different NVMe mounted at the same path passes signal 1 and fails here.
#
# THE SMALLEST VERIFIED FILE IS CHOSEN, so this costs a sidecar read (a few bytes) in the normal
# case rather than 85 GB. A prefix hash was the first design and it verified nothing: there is no
# recorded prefix digest to compare it to, so it would have hashed 8 MB and asserted that the
# result equalled itself.
if not os.path.exists(record):
    print("NO_RECORD " + record); sys.exit(3)
with open(record) as f:
    rec = json.load(f)
cands = []
for stem, g in sorted((rec.get("groups") or {}).items()):
    if not g.get("verified"):
        continue
    for name, s in (g.get("sha256") or {}).items():
        p = os.path.join(target, name)
        if s.get("dst") and s.get("bytes") and os.path.exists(p):
            cands.append((s["bytes"], name, p, s["dst"]))
if not cands:
    print("NO_VERIFIED_FILE no verified copy from the record is present at " + target); sys.exit(3)
nbytes, name, p, want = min(cands)
h = hashlib.sha256()
with open(p, "rb") as f:
    while True:
        b = f.read(8 << 20)
        if not b:
            break
        h.update(b)
got = h.hexdigest()
if got != want:
    print("SHA_MISMATCH %s: recorded %s, read %s -- something is mounted here, but it is not the "
          "filesystem the caches were verified on" % (name, want[:16], got[:16]))
    sys.exit(4)
print("MOUNTED dev=%d verified=%s bytes=%d sha=%s" % (os.stat(target).st_dev, name, nbytes, got[:16]))
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host-src", default="/data02/aupai",
                    help="host directory to attach (its tokens/ holds the caches)")
    ap.add_argument("--target", default="/mnt/data02",
                    help="path inside the container to attach it at")
    ap.add_argument("--record", default="/work/aupai/runs/token_cache_move.json",
                    help="the copy record, read INSIDE the container, for the sha check")
    ap.add_argument("--container", default=None, help="container id prefix, if several are running")
    ap.add_argument("--verify-only", action="store_true", help="check, never mount")
    a = ap.parse_args()

    pid, err = container_pid(a.container)
    if pid is None:
        print(f"REFUSING: {err}", file=sys.stderr)
        return 1
    print(f"container pid {pid}")

    tokens = os.path.join(a.target, "tokens")
    rc, out, errout = _verify(pid, tokens, a.record)
    print(f"verify: {out or errout}")
    if rc == 0:
        print("OK: already mounted and readable; nothing to do (idempotent)")
        return 0
    if a.verify_only:
        print(f"REFUSING: not mounted and --verify-only was given (rc={rc})", file=sys.stderr)
        return 1

    if not os.path.isdir(a.host_src):
        print(f"REFUSING: {a.host_src} is not a directory in the HOST namespace -- this script must "
              f"run on the host (tn exec), not inside the container", file=sys.stderr)
        return 1
    # The target must exist inside the container before move_mount; create it there.
    mk = subprocess.run(["nsenter", "-t", str(pid), "-m", "mkdir", "-p", a.target],
                        capture_output=True, text=True)
    if mk.returncode != 0:
        print(f"REFUSING: could not create {a.target} in the container: {mk.stderr.strip()[:160]}",
              file=sys.stderr)
        return 1
    r = subprocess.run([sys.executable, os.path.join(HERE, "host_mount_into_container.py"),
                        a.host_src, str(pid), a.target], capture_output=True, text=True)
    print(r.stdout.strip() or r.stderr.strip())
    if r.returncode != 0:
        return 1

    rc, out, errout = _verify(pid, tokens, a.record)
    print(f"verify after mount: {out or errout}")
    if rc != 0:
        print(f"REFUSING: the mount was attached and the verification still fails (rc={rc}). The "
              f"caches are NOT readable at {tokens}; do not launch a run against it.",
              file=sys.stderr)
        return 1
    print("OK: mounted and verified by reading")
    return 0


def _verify(pid, tokens, record):
    """The VERIFY snippet with its two paths substituted in as literals.

    Substitution rather than sys.argv: `nsenter -t <pid> -m python3 -c <src> a b` does pass the
    trailing words as argv, but the snippet then has to survive one more layer of quoting for every
    caller, and repr() of the two paths is exact. There is no shell in the chain (subprocess with a
    list), so nothing else re-parses it.
    """
    return in_container(pid, VERIFY.replace("TARGET", repr(tokens)).replace("RECORD", repr(record)))


if __name__ == "__main__":
    sys.exit(main())
