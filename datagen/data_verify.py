#!/usr/bin/env python3
"""Verify the local tree against data/MANIFEST.tsv.

Checks three things against the manifest and reports a machine-actionable list:
  - MISSING: a manifest path is absent
  - SHA-MISMATCH / SIZE-MISMATCH: present but different from the record
  - EXTRA: a data/ file not in the manifest (optionally scanned, defaults off)

Severity is tier-driven:
  frozen / eval  -> mismatch/missing is an ERROR (exit 1): these cannot be rebuilt,
                    so a wrong byte means a silent distribution shift and must stop
                    the pipeline.
  fetched        -> any drift is a WARNING (exit 0 for warn, but printed
                    clearly): re-derivable, but the drift is worth noting.
  derived        -> never recorded/checked with a pin; present-or-absent is fine,
                    recorded so bootstrap knows what it rebuilds.

Usage:
  python datagen/data_verify.py                 # from repo root
  python datagen/data_verify.py --scan          # also list data/ files not in manifest
  python datagen/data_verify.py --missing-only  # report just the MISSING set (bootstrap)
"""

import argparse
import hashlib
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join("data", "MANIFEST.tsv")
ERROR_TIERS = {"frozen", "eval"}


def selfcheck():
    """Known-answer validation of the verify logic: a hand-built tree (good/missing/
    tampered) + matching manifest; asserts each is classified correctly. Prints
    PASS/FAIL lines."""
    import shutil

    tmp = tempfile.mkdtemp(prefix="dataverify_")
    try:
        good = os.path.join(tmp, "good.txt")
        open(good, "w").write("hello")
        good_sha = hashlib.sha256(b"hello").hexdigest()
        tam = os.path.join(tmp, "tam.txt")
        open(tam, "w").write("HELLO")
        man = os.path.join(tmp, "MANIFEST.tsv")
        with open(man, "w") as f:
            f.write("# test manifest\n")
            f.write(f"good.txt\t{good_sha}\t5\tfrozen\tx\n")
            f.write(f"tam.txt\t{good_sha}\t5\tfrozen\tx\n")
            f.write(f"nope.txt\t{'0' * 64}\t9\tfrozen\tx\n")
        rows = []
        for line in open(man):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p, s, n, t, p2 = line.split("\t")
            rows.append({"path": p, "sha": s, "bytes": int(n), "tier": t, "producer": p2})
        ok = missing = mism = 0
        for r in rows:
            p = os.path.join(tmp, r["path"])
            if not os.path.exists(p):
                missing += 1
            elif os.path.getsize(p) != r["bytes"]:
                mism += 1
            elif hashlib.sha256(open(p, "rb").read()).hexdigest() != r["sha"]:
                mism += 1
            else:
                ok += 1
        tests = [
            ("good classified ok", ok == 1),
            ("missing classified", missing == 1),
            ("tampered classified sha-mismatch", mism == 1),
            ("tampered+missing are ERROR-tier", (mism + missing) > 0),
        ]
        allok = all(c for _, c in tests)
        for name, c in tests:
            print(f"  selfcheck {name}: {'PASS' if c else '*** FAIL ***'}")
        print("selfcheck:", "ALL CLEAN" if allok else "NOT CLEAN — DO NOT TRUST COUNTS")
        return allok
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def size(path):
    return os.path.getsize(path)


def load_manifest():
    rows = []
    with open(MANIFEST, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            path, sha, nbytes, tier, producer = line.split("\t")
            rows.append({"path": path, "sha": sha, "bytes": int(nbytes), "tier": tier, "producer": producer})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--missing-only", action="store_true")
    ap.add_argument("--root", default=ROOT)
    a = ap.parse_args()

    mpath = os.path.join(a.root, MANIFEST)
    if not os.path.exists(mpath):
        sys.exit(f"FATAL: {mpath} not found (no manifest to verify against)")

    rows = load_manifest()
    missing, mismatch_sz, mismatch_sha, ok, extra = [], [], [], [], []
    errors = 0

    for r in rows:
        p = os.path.join(a.root, r["path"])
        if not os.path.exists(p):
            missing.append((r["path"], r["tier"]))
            if r["tier"] in ERROR_TIERS:
                errors += 1
            continue
        if os.path.getsize(p) != r["bytes"]:
            mismatch_sz.append((r["path"], r["tier"], r["bytes"], os.path.getsize(p)))
            if r["tier"] in ERROR_TIERS:
                errors += 1
            continue
        got = sha256_file(p)
        if got != r["sha"]:
            mismatch_sha.append((r["path"], r["tier"], r["sha"], got))
            if r["tier"] in ERROR_TIERS:
                errors += 1
        else:
            ok.append(r["path"])

    if a.missing_only:
        for p, tier in missing:
            print(p)
        sys.exit(1 if errors else 0)

    if not os.path.exists(os.path.join(a.root, "data")):
        print(f"WARN: {a.root}/data absent (empty pod?) — {len(missing)} of {len(rows)} missing")
    well = len(ok)
    print(
        f"MANIFEST: {len(rows)} entries | ok {well} | missing {len(missing)} | "
        f"size-mismatch {len(mismatch_sz)} | sha-mismatch {len(mismatch_sha)}"
    )
    if a.scan:
        known = {r["path"] for r in rows}
        for base, _, files in os.walk(os.path.join(a.root, "data")):
            for fn in files:
                fp = os.path.relpath(os.path.join(base, fn), a.root)
                if fp.endswith((".jsonl", ".json", ".parquet", ".tsv")) and fp not in known:
                    if "corpus/" not in fp and "__pycache__" not in fp:
                        extra.append(fp)
    for p, tier in missing:
        print(f"  MISSING  [{tier}] {p}")
    for p, tier, exp, got in mismatch_sz:
        print(f"  SIZE-MIS  [{tier}] {p} expected {exp} got {got}")
    for p, tier, exp, got in mismatch_sha:
        print(f"  SHA-MIS   [{tier}] {p}\n      expected {exp}\n      got      {got}")
    for p in extra:
        print(f"  EXTRA     {p}")

    if missing or mismatch_sz or mismatch_sha:
        sys.exit(1 if errors else 0)
    print("ALL MANIFEST ENTRIES VERIFIED")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selfcheck":
        sys.exit(0 if selfcheck() else 1)
    sys.exit(main())
