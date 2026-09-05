#!/usr/bin/env python3
"""Copy the token caches from the overlay to NVMe, verifying every pair by sha256.

Report-only about /data00: NOTHING IS DELETED. Freeing the 231 GB is the user's call and it
happens, if at all, after this has been verified.

FOUR SIDECARS PER CACHE, NOT ONE. The task named tokens_*.pt and their .srcfp; the directory holds
.seed, .srcfp and .vocab as well -- 22 caches, 66 sidecars, 88 files. Copying only the .pt and the
.srcfp would leave a cache dir that assert_caches_fresh accepts (it reads .srcfp) while train.py's
shuffle seed and vocab_id are absent, which fails later and elsewhere. The set is discovered by
globbing tokens_* rather than by naming the extensions, so a fifth sidecar type is copied too.

SHA256 BOTH SIDES, and read back from the destination with O_DIRECT-ish behaviour by reading it
fresh after a sync. The reason to hash rather than trust the copy: the whole point of the move is
that a 231 GB read is slow, so a silent truncation or a torn block is exactly the failure this
cannot afford to discover during a training run three days from now.

--one copies the first cache only, which is how the rate gets measured before committing 20 minutes.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

CHUNK = 8 << 20


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(CHUNK)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def groups(src):
    """{stem: [files]} for every tokens_* file, grouped by the .pt they belong to."""
    out = {}
    for name in sorted(os.listdir(src)):
        if not name.startswith("tokens_"):
            continue
        stem = name.split(".pt")[0] + ".pt"
        out.setdefault(stem, []).append(name)
    return out


def free_bytes(path):
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize


def _nvme_default():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(root, "eval"))
    import cache_guard
    return cache_guard.NVME_CACHE_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=None,
                    help="default: train._token_cache_dir(), i.e. wherever the caches are read "
                         "from today. Naming it explicitly is how you copy FROM the overlay after "
                         "AUPAI_TOKEN_CACHE_DIR already points at NVMe")
    ap.add_argument("--dst", default=_nvme_default(),
                    help="default: eval/cache_guard.NVME_CACHE_DIR, the one definition every "
                         "accessor reads. A literal here would be a second copy, and the next "
                         "move would leave the two disagreeing about where the caches are")
    ap.add_argument("--one", action="store_true", help="copy one cache group and stop")
    ap.add_argument("--out", default="runs/token_cache_move.json")
    ap.add_argument(
        "--skip-done",
        action="store_true",
        help="skip a group this run's own record file already marks verified. It trusts the "
        "record and does NOT re-hash: use it to resume a killed run, never to audit one.",
    )
    a = ap.parse_args()

    if a.src is None:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        import train

        a.src = train._token_cache_dir()
        print(f"src {a.src} (train._token_cache_dir)")

    # THE DESTINATION MUST NOT BE THE SOURCE'S FILESYSTEM. That is the whole point, and it is
    # exactly the mistake the container's /data00 naming invites: a move_mount that silently did
    # not take effect leaves the dst on the overlay, the copy "succeeds", every sha matches, and
    # 231 GB has been duplicated onto the disk that was already 87% full.
    os.makedirs(a.dst, exist_ok=True)
    if os.stat(a.dst).st_dev == os.stat(a.src).st_dev:
        print(
            f"REFUSING: {a.dst} and {a.src} are on the SAME filesystem "
            f"(dev {os.stat(a.src).st_dev}). The NVMe mount is not in effect -- copying now "
            f"would duplicate the caches onto the overlay that is already 87% full.",
            file=sys.stderr,
        )
        return 1

    g = groups(a.src)
    if not g:
        print(f"REFUSING: no tokens_* files under {a.src}", file=sys.stderr)
        return 1
    total = sum(os.path.getsize(os.path.join(a.src, n)) for names in g.values() for n in names)
    print(f"{len(g)} cache group(s), {sum(len(v) for v in g.values())} files, {total / 1e9:.1f} GB")
    avail = free_bytes(a.dst)
    print(f"destination free: {avail / 1e9:.1f} GB")
    if not a.one and avail < total * 1.02:
        print(
            f"REFUSING: destination has {avail / 1e9:.1f} GB free, needs {total / 1e9:.1f} GB",
            file=sys.stderr,
        )
        return 1

    rec = {"src": a.src, "dst": a.dst, "groups": {}}
    if os.path.exists(a.out):
        try:
            with open(a.out) as f:
                rec = json.load(f)
            rec.setdefault("groups", {})
        except (OSError, ValueError):
            pass

    order = list(g)
    if a.one:
        # The SMALLEST group first when measuring, so the rate number costs seconds not minutes.
        order = [min(g, key=lambda s: os.path.getsize(os.path.join(a.src, s)))]

    for stem in order:
        names = g[stem]
        gbytes = sum(os.path.getsize(os.path.join(a.src, n)) for n in names)
        if a.skip_done and rec["groups"].get(stem, {}).get("verified"):
            print(f"skip  {stem}  (already verified)")
            continue
        t0 = time.time()
        pairs = []
        ok = True
        for n in names:
            s, d = os.path.join(a.src, n), os.path.join(a.dst, n)
            shutil.copyfile(s, d)
            pairs.append((n, s, d))
        subprocess.run(["sync"], check=False)
        copy_s = time.time() - t0

        # HASH AFTER THE SYNC, and hash the DESTINATION by reading it back rather than reusing
        # anything computed during the write.
        t1 = time.time()
        shas = {}
        for n, s, d in pairs:
            hs, hd = sha256(s), sha256(d)
            shas[n] = {"src": hs, "dst": hd, "bytes": os.path.getsize(s), "match": hs == hd}
            if hs != hd:
                ok = False
                print(f"  MISMATCH {n}: src {hs[:16]} dst {hd[:16]}", file=sys.stderr)
        hash_s = time.time() - t1
        rec["groups"][stem] = {
            "files": len(names),
            "bytes": gbytes,
            "copy_seconds": round(copy_s, 1),
            "copy_MB_per_s": round(gbytes / 1e6 / copy_s, 1) if copy_s else None,
            "hash_seconds": round(hash_s, 1),
            "sha256": shas,
            "verified": ok,
        }
        # ATOMIC PER GROUP, so a killed run leaves every completed group recorded rather than
        # nothing (the corpus scan needed the same fix).
        tmp = a.out + ".tmp"
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(tmp, "w") as f:
            json.dump(rec, f, indent=1)
        os.replace(tmp, a.out)
        print(
            f"{'OK  ' if ok else 'FAIL'}  {stem}  {gbytes / 1e9:.2f} GB  "
            f"copy {copy_s:.1f}s ({gbytes / 1e6 / copy_s:.0f} MB/s)  hash {hash_s:.1f}s"
        )
        if not ok:
            print("REFUSING to continue after a sha mismatch", file=sys.stderr)
            return 1

    done = [s for s, v in rec["groups"].items() if v.get("verified")]
    print(f"verified {len(done)} of {len(g)} group(s); nothing deleted from {a.src}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
