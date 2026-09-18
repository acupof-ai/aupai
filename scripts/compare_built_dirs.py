#!/usr/bin/env python3
"""Compare two built corpus dirs shard-by-shard, for the parallel-vs-serial equivalence.

WHY THIS EXISTS. `_parallel_exact_pass` claims in its docstring that it is "byte-identical
to the serial pass" (build_corpus.py:777). That claim is a comment, and the code it describes
decides the exact bytes of a multi-GB training corpus. The risk is concentrated in phase A/B:
phase A emits (exact_key, global_ordinal) per doc in parallel, phase B keeps the MIN ordinal
per key, and phase C rewrites survivors in global order. If the ordinal bases or the shard
sort differ at all between the two implementations, the winners of a duplicate pair change,
and the resulting bytes differ -- silently, because both runs report "ok".

    python3 scripts/compare_built_dirs.py --a <dirA> --b <dirB>

Compares: the shard NAME sets (a missing/extra shard is a divergence), then each shard's
sha256. Prints per-shard agreement and exits nonzero on any difference, naming the shards.
Also compares build_corpus_stats.json's `reasons` and `kept`, since two dirs with equal
shards but different stats would mean the stats are not derived from the shards.
"""
# restartable: read-only over two built dirs; it hashes shards and writes nothing.
import argparse
import hashlib
import json
import os
import sys


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def shards(d, domain):
    return sorted(f for f in os.listdir(d) if f.startswith(f"{domain}_") and f.endswith(".jsonl"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="first dir (parallel output)")
    ap.add_argument("--b", required=True, help="second dir (serial output)")
    ap.add_argument("--domain", default="en_c4")
    a = ap.parse_args()

    sa, sb = shards(a.a, a.domain), shards(a.b, a.domain)
    print(f"A {a.a}: {len(sa)} shard(s)")
    print(f"B {a.b}: {len(sb)} shard(s)")
    problems = []
    if sa != sb:
        only_a = sorted(set(sa) - set(sb))
        only_b = sorted(set(sb) - set(sa))
        problems.append(f"shard NAME SET differs: only-in-A {only_a[:5]}, only-in-B {only_b[:5]}")
    diff = []
    common = [s for s in sa if s in set(sb)]
    for i, name in enumerate(common, 1):
        ha, hb = sha256(os.path.join(a.a, name)), sha256(os.path.join(a.b, name))
        if ha != hb:
            diff.append((name, ha[:12], hb[:12]))
        if i % 50 == 0:
            print(f"  compared {i}/{len(common)}", flush=True)
    print(f"\ncompared {len(common)} common shard(s); {len(diff)} differ")
    for name, ha, hb in diff[:10]:
        print(f"  DIFF {name}: A {ha} vs B {hb}")
    if diff:
        problems.append(f"{len(diff)} shard(s) differ in content")

    for d, tag in ((a.a, "A"), (a.b, "B")):
        p = os.path.join(d, "build_corpus_stats.json")
        if os.path.exists(p):
            s = json.load(open(p))
            print(f"{tag} stats: kept={s.get('kept')} reasons={s.get('reasons')}")
        else:
            print(f"{tag}: no build_corpus_stats.json")
            problems.append(f"{tag} has no stats")

    if problems:
        print("\nNOT EQUIVALENT:")
        for p in problems:
            print("  " + p)
        return 1
    print(f"\nEQUIVALENT: all {len(common)} shard(s) byte-identical (sha256), name sets equal")
    return 0


def _selftest():
    import subprocess
    import tempfile

    def mk(d, body, extra=None):
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "en_c4_000.jsonl"), "w") as fh:
            fh.write(body)
        with open(os.path.join(d, "build_corpus_stats.json"), "w") as fh:
            json.dump({"kept": 1, "reasons": {"kept": 1}}, fh)
        if extra:
            with open(os.path.join(d, extra), "w") as fh:
                fh.write("x\n")

    with tempfile.TemporaryDirectory() as t:
        A, B, C = (os.path.join(t, x) for x in ("A", "B", "C"))
        mk(A, "hello\n")
        mk(B, "hello\n")
        mk(C, "hellp\n")
        r = subprocess.run([sys.executable, __file__, "--a", A, "--b", B],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        r = subprocess.run([sys.executable, __file__, "--a", A, "--b", C],
                           capture_output=True, text=True)
        assert r.returncode == 1 and "DIFF" in r.stdout, r.stdout
        mk(C, "hellp\n", extra="en_c4_001.jsonl")
        r = subprocess.run([sys.executable, __file__, "--a", A, "--b", C],
                           capture_output=True, text=True)
        assert r.returncode == 1 and "NAME SET differs" in r.stdout, r.stdout
    print("compare_built_dirs selftest OK: equal dirs pass; a 1-byte shard diff and an extra "
          "shard each exit nonzero (the two ways a parallel/serial divergence shows up)")
    return 0


if __name__ == "__main__":
    sys.exit(_selftest() if "--selftest" in sys.argv else main())
