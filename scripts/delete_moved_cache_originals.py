#!/usr/bin/env python3
"""Delete the /data00 token cache originals, by exact path, only where the NVMe copy verifies.

User order relayed by 4c, 2026-09-05: the /data00 originals may be deleted. Named target: exactly
the 88 files listed in runs/token_cache_move.json (22 caches x 4 sidecars, 247.8 GB), nothing else
under /data00.

THE SAFEGUARD IS THE RULE, NOT A CAUTION (4c, approved): a file is deleted only if it is IN the
record AND its NVMe copy still hashes to the recorded digest. A group whose copy does not verify
keeps its original and is reported. Two independent reasons: the copy was verified once at copy
time and a mount can be remounted or a file truncated since, and a record can name a file that was
later replaced. Neither is caught by the path existing.

NO GLOB AND NO DIRECTORY REMOVAL. Every unlink takes a path built from a record key, and the
parent directory is never touched. A path that does not start with the record's own `src` is
refused rather than skipped -- that would mean the record disagrees with itself.

--apply performs the deletion; without it this is a dry run that prints exactly what would go.
Default is the dry run because the reverse default deletes 247.8 GB on a typo.
"""
# restartable: an interrupt is cheap and rerunning is safe by construction, not by luck. Phase 1
# verifies every NVMe copy BEFORE phase 2 unlinks anything, so a kill during verification has
# deleted nothing at all. A kill mid-unlink leaves the remaining originals in place, and a rerun
# re-verifies from the record and skips any source already gone (`if not os.path.exists(s):
# continue`), so it deletes exactly the remainder. The only work lost is the hashing done so far,
# about 7 minutes of NVMe reads. The record is written tmp-then-move, so a kill during the write
# leaves the previous record intact rather than a truncated one; the cost is that a kill between
# the last unlink and the move loses that run's DELETION LOG, and the rerun then reports those
# files as already gone rather than as deleted by it.

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECORD = os.path.join(ROOT, "runs", "token_cache_move.json")


def sha256_file(p, chunk=8 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def df(path):
    """Bytes used/free and the percentage, from statvfs -- not by parsing df's text output."""
    try:
        s = os.statvfs(path)
    except OSError:
        return None
    total = s.f_blocks * s.f_frsize
    free = s.f_bavail * s.f_frsize
    used = total - (s.f_bfree * s.f_frsize)
    return {"total": total, "used": used, "avail": free,
            "pct": round(100.0 * used / total, 1) if total else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="actually delete (default: dry run)")
    ap.add_argument("--record", default=RECORD)
    args = ap.parse_args()

    if not os.path.exists(args.record):
        print(f"REFUSING: no record at {args.record}")
        return 1
    with open(args.record) as fh:
        rec = json.load(fh)
    src_root = rec.get("src")
    dst_root = rec.get("dst")
    groups = rec.get("groups")
    if not src_root or not dst_root or not isinstance(groups, dict) or not groups:
        print(f"REFUSING: record is missing src/dst/groups; keys are {sorted(rec)}")
        return 1
    print(f"record: src={src_root} dst={dst_root} groups={len(groups)}")

    df_before = {"src": df(src_root), "dst": df(dst_root)}
    for k, v in df_before.items():
        if v:
            print(f"df before  {k:<4} {v['used'] / 1e9:>9.1f} GB used  {v['avail'] / 1e9:>9.1f} GB "
                  f"avail  {v['pct']}%")

    # PHASE 1: verify every NVMe copy. Nothing is deleted until the whole set is judged, so a
    # failure in group 22 does not leave groups 1-21 already gone.
    plan, refused = [], []
    for cache_name in sorted(groups):
        g = groups[cache_name]
        shas = g.get("sha256") or {}
        if not shas:
            refused.append((cache_name, "the record holds no sha256 map for this group"))
            continue
        group_plan, why = [], []
        for fname, entry in sorted(shas.items()):
            s = os.path.join(src_root, fname)
            d = os.path.join(dst_root, fname)
            want = entry.get("dst")
            if not want:
                why.append(f"{fname}: no dst digest recorded")
                continue
            if os.path.realpath(s) != os.path.normpath(os.path.join(src_root, fname)):
                why.append(f"{fname}: source path escapes the record's src root")
                continue
            if not os.path.exists(d):
                why.append(f"{fname}: NVMe copy ABSENT")
                continue
            got = sha256_file(d)
            if got != want:
                why.append(f"{fname}: NVMe digest {got[:16]} != recorded {want[:16]}")
                continue
            if not os.path.exists(s):
                continue  # already gone; not an error, and nothing to delete
            group_plan.append((s, os.path.getsize(s)))
        if why:
            refused.append((cache_name, "; ".join(why)))
            print(f"  KEEP     {cache_name:<30} {why[0]}")
        else:
            plan += group_plan
            print(f"  verified {cache_name:<30} {len(group_plan)} original(s) to delete, "
                  f"{sum(n for _, n in group_plan) / 1e9:.2f} GB")

    total = sum(n for _, n in plan)
    print(f"\n{len(plan)} file(s), {total / 1e9:.1f} GB to delete; {len(refused)} group(s) KEPT")
    if refused:
        print("KEPT groups (their originals stay):")
        for name, why in refused:
            print(f"  {name}: {why}")

    if not args.apply:
        print("\nDRY RUN. Nothing deleted. Re-run with --apply.")
        return 0
    if not plan:
        print("nothing to delete")
        return 0

    # PHASE 2: delete by exact path, logging each one and its size.
    deleted, failed = [], []
    for p, n in plan:
        try:
            os.unlink(p)
            deleted.append({"path": p, "bytes": n})
            print(f"  deleted {p}  {n / 1e9:.2f} GB")
        except OSError as e:
            failed.append({"path": p, "error": str(e)})
            print(f"  FAILED  {p}: {e}")

    subprocess.run(["sync"], check=False)
    df_after = {"src": df(src_root), "dst": df(dst_root)}
    for k, v in df_after.items():
        if v:
            print(f"df after   {k:<4} {v['used'] / 1e9:>9.1f} GB used  {v['avail'] / 1e9:>9.1f} GB "
                  f"avail  {v['pct']}%")

    freed = None
    if df_before["src"] and df_after["src"]:
        freed = df_before["src"]["used"] - df_after["src"]["used"]
        print(f"\nfreed on {src_root}: {freed / 1e9:.1f} GB "
              f"({df_before['src']['pct']}% -> {df_after['src']['pct']}%)")
        print(f"sum of deleted file sizes: {sum(d['bytes'] for d in deleted) / 1e9:.1f} GB")
        if freed < 0.9 * sum(d["bytes"] for d in deleted):
            print("  NOTE: freed is well below the deleted bytes. A process still holding an open "
                  "fd on a deleted file keeps its blocks until it exits, so this is expected if a "
                  "run was reading a cache -- it is NOT a failed delete.")

    # Record it, in the same file, atomically.
    rec["deleted"] = {
        "when": subprocess.run(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], capture_output=True,
                               text=True).stdout.strip(),
        "files": deleted,
        "failed": failed,
        "kept_groups": {name: why for name, why in refused},
        "df_before": df_before,
        "df_after": df_after,
        "freed_bytes": freed,
    }
    tmp = args.record + ".tmp"
    with open(tmp, "w") as f:
        json.dump(rec, f, indent=1)
    shutil.move(tmp, args.record)
    print(f"recorded in {args.record}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
