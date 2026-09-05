#!/usr/bin/env python3
"""Cardless: does every cache in the move record resolve, through train's accessor, to a file on
NVMe -- and does that file still hash to what the record says?

4c's step (2), and the gate before any deletion. Two properties, not one:

  RESOLVE   for each of the 22 domains, train._domain_cache_path(domain) points at an existing
            file under the NVMe dir. This is what makes the copy REACHABLE by a run: a verified
            file the accessor does not resolve to is not in use by anything.
  MATCH     that file's sha256 equals the `dst` digest the move recorded. This is what makes the
            deletion SAFE. The copy was verified once, at copy time; between then and now the
            mount could have been remounted, the file truncated by a full disk, or the record
            could name a file that was replaced.

The digest half is why this is not just an ls. A path that exists proves the name resolves; it
does not prove the bytes are the ones that were verified.

NO CARD AND NO CACHE LOAD. It imports train for the accessor, stats every file, and hashes the
NVMe copies. Hashing 247.8 GB off NVMe is ~7 min of read at 1.3 GB/s and touches no /data00 byte,
so it does not contend for the overlay the way a torch.load of a cache would (AGENTS: an eval that
reads a token cache off /data00 waits for the run; this reads the NVMe side only).

--quick hashes only the SMALLEST file per group instead of all four: enough to prove the mount is
live and the group is present, not enough to authorise a deletion. It prints which mode it ran in
and the exit code differs in neither -- the caller must read the mode, so the delete script takes
the full mode's record and nothing else.
"""
import hashlib
import json
import os
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


def main():
    quick = "--quick" in sys.argv
    if not os.path.exists(RECORD):
        print(f"REFUSING: no move record at {RECORD} -- nothing to verify against")
        return 1
    rec = json.load(open(RECORD))
    groups = rec.get("groups")
    if not isinstance(groups, dict) or not groups:
        print(f"REFUSING: record's `groups` is {type(groups).__name__}, expected a non-empty dict")
        return 1

    os.environ.setdefault("AUPAI_TOKEN_CACHE_DIR", rec.get("dst", "/mnt/data02/tokens"))
    sys.path.insert(0, ROOT)
    import train

    cache_dir = train._token_cache_dir()
    print(f"record src={rec.get('src')} dst={rec.get('dst')}  groups={len(groups)}")
    print(f"accessor cache dir: {cache_dir}   AUPAI_TOKEN_CACHE_DIR={os.environ['AUPAI_TOKEN_CACHE_DIR']}")
    print(f"Cfg.fone={train.Cfg.fone}   mode={'QUICK (not sufficient for deletion)' if quick else 'FULL'}\n")

    resolve_bad, digest_bad, verified = [], [], []
    for cache_name in sorted(groups):
        g = groups[cache_name]
        # The record's key is the cache FILENAME; the accessor takes a DOMAIN. Derive the domain
        # from the filename and then assert the accessor round-trips back to the same name -- that
        # is the join being tested, and deriving it the other way would assume it.
        if not cache_name.startswith("tokens_") or not cache_name.endswith(".pt"):
            resolve_bad.append((cache_name, f"unexpected record key shape: {cache_name}"))
            continue
        domain = cache_name[len("tokens_"):-len(".pt")]
        if domain.endswith("_fone"):
            domain = domain[:-len("_fone")]
        p = train._domain_cache_path(domain)
        if os.path.basename(p) != cache_name:
            resolve_bad.append((cache_name, f"accessor produced {os.path.basename(p)} "
                                            f"(Cfg.fone={train.Cfg.fone}); the names disagree"))
            continue
        if os.path.realpath(os.path.dirname(p)) != os.path.realpath(cache_dir):
            resolve_bad.append((cache_name, f"resolves outside the cache dir: {p}"))
            continue

        shas = g.get("sha256") or {}
        to_hash = sorted(shas, key=lambda k: shas[k].get("bytes", 0))
        if quick:
            to_hash = to_hash[:1]
        group_ok, group_bad = [], []
        for fname in to_hash:
            fp = os.path.join(cache_dir, fname)
            want = shas[fname].get("dst")
            if not os.path.exists(fp):
                group_bad.append(f"{fname}: ABSENT on NVMe")
                continue
            got = sha256_file(fp)
            if got != want:
                group_bad.append(f"{fname}: digest {got[:16]} != recorded {str(want)[:16]}")
            else:
                group_ok.append(fname)
        if group_bad:
            digest_bad.append((cache_name, group_bad))
            print(f"  BAD      {cache_name:<30} " + "; ".join(group_bad[:2]))
        else:
            verified.append((cache_name, len(group_ok), g.get("bytes", 0)))
            print(f"  verified {cache_name:<30} {len(group_ok)}/{len(shas)} file(s) hashed, "
                  f"{g.get('bytes', 0) / 1e9:.2f} GB")

    for name, why in resolve_bad:
        print(f"  RESOLVE  {name:<30} {why}")

    total = sum(b for _, _, b in verified)
    print(f"\n{len(verified)}/{len(groups)} groups verified, {total / 1e9:.1f} GB")
    if resolve_bad or digest_bad:
        print(f"REFUSING: {len(resolve_bad)} name(s) do not resolve and {len(digest_bad)} group(s) "
              f"do not match their recorded digest. The /data00 original for each of those must NOT "
              f"be deleted.")
        return 1
    if quick:
        print("QUICK mode: one file per group. This proves the mount is live and every name "
              "resolves. It does NOT authorise a deletion -- rerun without --quick for that.")
        return 0
    print("Every cache name resolves through the accessor to a file on NVMe whose sha256 still "
          "equals the digest recorded at copy time.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
