#!/usr/bin/env python3
"""Fetch a named corpus source to data/raw/<source>/ with shard-level resumability.

The corpus-half fetch step under `harness run fetch --source <name> [--target_bytes N]`.
de owns the harness wrapper; this script owns the substance.

Contract (from de/fb): shard-level resumability (per-shard write, skip completed,
resume from the first incomplete -- the restartability audit passes because each
shard's write is inside the per-shard loop), a `source_fp` content-hash fingerprint
of the actual source manifest (URLs + sizes/etags per shard -- a re-fetch from a
changed upstream gives a different fp), the disk guard (`data/raw` free >=
target_bytes * 1.5 AND not on the container overlay -- an overlay reports the
free bytes of its backing fs but is wiped on restart, so free space alone is
not a guard), and data/raw as a real dir on /work (never a symlink to an
overlay). Exit 0 on success, non-zero otherwise.

    python scripts/fetch_corpus.py --source fineweb2 --target_bytes 30e9
    python scripts/fetch_corpus.py --source cci3_hq --target_bytes 50e9
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
CHUNK = 1 << 20


def ensure_raw_location():
    """data/raw is a real directory on the container's disk (/work), never a
    symlink to an overlay layer. /data00 is NOT a disk: it is the container's
    overlay root (same st_dev as /), wiped on restart — it swallowed the
    token-cache once already with no error. 36B raw fits /work if measured
    before pulling (900G free); if /work cannot hold it, that is a refusal,
    not a relocation to an overlay."""
    os.makedirs(RAW, exist_ok=True)


def _on_overlay(path):
    """True if path lives on the container overlay (same device as /), which
    disappears on restart. The st_dev check is what free-space never catches:
    an overlay reports the free bytes of its backing file system, so a guard
    that only checks capacity approves 260GB on a layer that will vanish."""
    root_dev = os.stat("/").st_dev
    return os.stat(path).st_dev == root_dev


def disk_ok(target_bytes):
    if _on_overlay(RAW):
        print(
            f"REFUSING: data/raw ({RAW}) is on the container overlay (st_dev "
            f"{os.stat(RAW).st_dev} == root's {os.stat('/').st_dev}), which is "
            f"wiped on restart. Move data/raw onto a real disk (/work) first.",
            file=sys.stderr,
        )
        return False
    free = shutil.disk_usage(RAW).free
    need = target_bytes * 1.5
    ok = free >= need
    print(
        f"data/raw free {free / 1e9:.1f}G vs target*1.5 {need / 1e9:.1f}G -> "
        f"{'ok' if ok else 'REFUSE'}"
    )
    return ok


def _sha1(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(CHUNK), b""):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------- sources
# Each source resolves to a manifest: list of (relative_shard, url, expected_bytes).
# The manifest IS the fetch recipe; source_fp = content hash of it, so a changed
# upstream (new/renamed shard, new size) changes the fingerprint.
def _manifest_fineweb2():
    import urllib.request

    base = "https://hf-mirror.com/datasets/HuggingFaceFW/fineweb-2/resolve/main/data/cmn_Hani/train"
    api = "https://hf-mirror.com/api/datasets/HuggingFaceFW/fineweb-2/tree/main/data/cmn_Hani/train"
    with urllib.request.urlopen(api, timeout=30) as r:
        rows = json.loads(r.read())
    man = []
    for row in rows:
        path = row.get("path", "")
        if path.endswith(".parquet"):
            name = path.split("/")[-1]
            man.append((name, f"{base}/{name}", int(row.get("size", 0))))
    return man


def _manifest_cci3_hq():
    import urllib.request

    api = (
        "https://www.modelscope.cn/api/v1/datasets/BAAI/CCI3-HQ/repo/tree"
        "?Revision=master&Root=&Recursive=true"
    )
    with urllib.request.urlopen(api, timeout=30) as r:
        d = json.loads(r.read())
    base = "https://www.modelscope.cn/api/v1/datasets/BAAI/CCI3-HQ/repo?Revision=master&FilePath="
    man = []
    for f in d["Data"]["Files"]:
        if f.get("Type") == "blob" and f["Path"].startswith("data/"):
            man.append((f["Path"].split("/")[-1], base + f["Path"], int(f.get("Size", 0))))
    return man


def _manifest_rp1t_github():
    """RedPajama-1T github slice : ``filtered_<sha>.sampled.jsonl`` on
    data.together.xyz (uncompressed jsonl, 2.66GB each; ~0.283 tok/byte frozen-
    vocab exact). The manifest is the shipped names file; its content hash is the
    source_fp. Reachable ONLY via IPv4 (the pod's IPv6 egress is broken -> curl -4).
    """
    names = open(os.path.join(ROOT, "data", "raw", "rp1t_github_manifest.txt")).read().split()
    base = "https://data.together.xyz/redpajama-data-1T/v1.0.0/github/"
    return [(n, base + n, 0) for n in names]


def _manifest_fineweb_edu_10bt():
    """en cell (5.5B): fineweb-edu sample/10BT -- 14 parquet (~28.5GB, >10B GPT-2
    tokens; edu-filtered, the selection we would hand-make for reasoning). File
    names are 000_00000.parquet..013_00000.parquet under sample/10BT/. Reachable
    via hf-mirror resolve with curl -4 (the pod's IPv6 egress is broken). The
    manifest is the shipped names file (source_fp = its content hash)."""
    names = open(os.path.join(ROOT, "data", "raw", "fineweb_edu_10bt_manifest.txt")).read().split()
    base = "https://hf-mirror.com/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/"
    return [(n, base + n, 0) for n in names]


def _manifest_rp1t_c4():
    """en cell (fb 2026-08-31, hf-mirror down): RedPajama-1T c4 slice on
    data.together.xyz (up while hf-mirror is rc=28). 1024 c4-train files, take in
    manifest order until ~27GB. Reachable via curl -4. Manifest = shipped names
    file (source_fp = its content hash)."""
    names = open(os.path.join(ROOT, "data", "raw", "rp1t_c4_manifest.txt")).read().split()
    base = "https://data.together.xyz/redpajama-data-1T/v1.0.0/c4/"
    return [(n, base + n, 0) for n in names]


def _manifest_rp1t_arxiv():
    """math role (en, feeds math until OpenWebMath reachable): RedPajama-1T arxiv
    slice on data.together (up while hf-mirror is down). 100 arxiv_*.jsonl."""
    names = open(os.path.join(ROOT, "data", "raw", "rp1t_arxiv_manifest.txt")).read().split()
    base = "https://data.together.xyz/redpajama-data-1T/v1.0.0/arxiv/"
    return [(n, base + n, 0) for n in names]


def _manifest_rp1t_stackexchange():
    """cot-role QA (code+math): RedPajama-1T stackexchange slice (single file)."""
    names = open(os.path.join(ROOT, "data", "raw", "rp1t_stackexchange_manifest.txt")).read().split()
    base = "https://data.together.xyz/redpajama-data-1T/v1.0.0/stackexchange/"
    return [(n, base + n, 0) for n in names]


def _manifest_ms_finemath_4plus():
    """math cell (real source, ModelScope up while hf-mirror down 2026-08-31):
    AI-ModelScope/finemath finemath-4plus, 64 parquet (HuggingFaceTB/finemath's
    filtered math-web set). Resolve via modelscope.cn/datasets/<org>/<name>/resolve/master/."""
    names = open(os.path.join(ROOT, "data", "raw", "ms_finemath_4plus_manifest.txt")).read().split()
    base = "https://www.modelscope.cn/datasets/AI-ModelScope/finemath/resolve/master/"
    return [(n, base + n, 0) for n in names]



def _manifest_ms_om2():
    """cot role (ModelScope, 2026-08-31): OpenMathInstruct-2, 55 data parquet
    (small, under the finemath large-file abort; direct, no chunking)."""
    names = open(os.path.join(ROOT, "data", "raw", "ms_om2_manifest.txt")).read().split()
    base = "https://www.modelscope.cn/datasets/AI-ModelScope/OpenMathInstruct-2/resolve/master/"
    return [(n, base + n, 0) for n in names]



def _manifest_hf_finemath_4plus():
    names = open(os.path.join(ROOT, "data", "raw", "hf_finemath_4plus_manifest.txt")).read().split()
    base = "https://hf-mirror.com/datasets/HuggingFaceTB/finemath/resolve/refs%2Fconvert%2Fparquet/"
    return [(n, base + n, 0) for n in names]


def _manifest_hf_om2():
    names = open(os.path.join(ROOT, "data", "raw", "ms_om2_manifest.txt")).read().split()
    base = "https://hf-mirror.com/datasets/open-math/OpenMathInstruct-2/resolve/main/"
    return [(n, base + n, 0) for n in names]


SOURCES = {
    "fineweb2": _manifest_fineweb2,
    "cci3_hq": _manifest_cci3_hq,
    "rp1t_github": _manifest_rp1t_github,
    "en_fineweb_edu": _manifest_fineweb_edu_10bt,
    "rp1t_c4": _manifest_rp1t_c4,
    "rp1t_arxiv": _manifest_rp1t_arxiv,
    "rp1t_stackexchange": _manifest_rp1t_stackexchange,
    "ms_finemath_4plus": _manifest_ms_finemath_4plus,
    "ms_om2": _manifest_ms_om2,
    "hf_finemath_4plus": _manifest_hf_finemath_4plus,
    "hf_om2": _manifest_hf_om2,
}


def source_fp(manifest):
    h = hashlib.sha1()
    for name, url, size in sorted(manifest):
        h.update(f"{name}\t{size}\t{url}\n".encode())
    return h.hexdigest()


# ---------------------------------------------------------------- fetch
def _refuse_prev_fp(source, source_fp):
    """A resume against a changed upstream must not silently mix two source states:
    refuse. Same failure class as a token cache reused against a swapped corpus."""
    for name in ("fetch_stats.json", "fetch_stats.log"):
        p = os.path.join(RAW, source, name)
        if not os.path.exists(p):
            continue
        try:
            prev = json.loads(open(p).read()) if name.endswith(".json") else None  # noqa: SIM115
        except Exception:
            prev = None
        if prev and prev.get("source_fp") and prev["source_fp"] != source_fp:
            return prev["source_fp"]
    return None


def fetch(source, target_bytes, stream_n=0, stream_i=0):
    ensure_raw_location()
    if not disk_ok(target_bytes):
        print(f"REFUSING: data/raw does not hold {target_bytes * 1.5 / 1e9:.1f}G needed", file=sys.stderr)
        return 2
    getter = SOURCES.get(source)
    if getter is None:
        print(f"unknown source {source!r}; known: {sorted(SOURCES)}", file=sys.stderr)
        return 2
    manifest = getter()
    if not manifest:
        print(
            f"source {source} resolved to an empty manifest -- upstream may be unreachable", file=sys.stderr
        )
        return 2
    outdir = os.path.join(RAW, source)
    os.makedirs(outdir, exist_ok=True)
    fp = source_fp(manifest)

    stale = _refuse_prev_fp(source, fp)
    if stale:
        # the prior fetch was a DIFFERENT source state: its .part files are a
        # third state (neither complete nor from this source) -- delete them
        # rather than trust. On a same-source resume (no stale), .part files are
        # RESUMABLE partials and must be kept for `-C -`.
        for part in sorted(os.path.join(outdir, x) for x in os.listdir(outdir) if x.endswith(".part")):
            os.remove(part)
            print(f"  removed stale .part {os.path.basename(part)}", file=sys.stderr)
        print(
            f"REFUSING: upstream changed since the prior fetch (recorded source_fp {stale} != "
            f"current {fp}). Shards from two source states would share one fingerprint. "
            f"Move trust and re-fetch, or reconcile deliberately.",
            file=sys.stderr,
        )
        return 4
    print(f"{source}: {len(manifest)} shards, source_fp {fp}")

    stats = {"source": source, "source_fp": fp, "target_bytes": target_bytes, "shards": []}
    log = os.path.join(outdir, "fetch_stats.log")  # per-shard append log: resume evidence
    got = 0
    for idx_, (name, url, expect) in enumerate(manifest):
        if stream_n and idx_ % stream_n != stream_i:
            continue
        if got >= target_bytes:
            break
        dst = os.path.join(outdir, name)
        part = dst + ".part"  # partial shards use temp name: rename on completion, atomic
        if os.path.exists(dst):  # a completed shard (final name exists) -- skip, but verify
            sz = os.path.getsize(dst)
            if expect and sz != expect:
                print(f"  {name}: final {sz}B != manifest {expect}B -- corrupt, re-fetch", file=sys.stderr)
                os.remove(dst)
            else:
                rec = {"shard": name, "bytes": sz, "status": "skipped-verified"}
                with open(log, "a") as f:  # incremental per-shard record
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                stats["shards"].append(rec)
                got += sz
                continue
        # periodic disk re-check: refuse the next shard rather than corrupt a full partition
        if shutil.disk_usage(RAW).free < target_bytes * 1.5:
            print(
                f"  disk low ({shutil.disk_usage(RAW).free / 1e9:.0f}G free); clean stop at {got / 1e9:.1f}G",
                file=sys.stderr,
            )
            break
        # write to temp name, then atomic rename: existence of dst == completeness.
        # curl must NOT be Check=True: a transient HTTP/2 stream error (exit 92,
        # seen 2026-08-30: it killed the whole 11-file fetch 12 hours in) nets
        # out to a nonzero curl exit even after its internal --retry 6. So we
        # retry the SAME shard (resuming the .part via -C -) with bounded outer
        # backoff, and only give up this shard loudly -- never crash the run.
        attempts = 0
        while True:
            r = subprocess.run(
                ["curl", "-4", "-sL", "-C", "-", "-o", part, "--retry", "6", "--retry-delay", "3", url],
                stdout=subprocess.DEVNULL,
            )
            if r.returncode == 0:
                break
            attempts += 1
            print(
                f"  {name}: curl exit {r.returncode} (stream/net err) attempt {attempts}; "
                f"resuming .part, backing off",
                file=sys.stderr,
                flush=True,
            )
            if attempts >= 4:
                print(
                    f"  {name}: giving up this shard after {attempts} outer retries; "
                    f".part kept ({os.path.getsize(part) if os.path.exists(part) else 0}B) for resume",
                    file=sys.stderr,
                    flush=True,
                )
                return 3
            import time

            time.sleep(20 * attempts)
        sz = os.path.getsize(part) if os.path.exists(part) else 0
        if expect and sz != expect:
            print(
                f"  {name}: {sz}B != expected {expect}B -- fetch incomplete, .part kept for resume",
                file=sys.stderr,
            )
            return 3
        os.rename(part, dst)  # atomic: a partial shard never masquerades as complete
        rec = {"shard": name, "bytes": sz, "status": "fetched"}
        with open(log, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        stats["shards"].append(rec)
        got += sz
        print(f"  {name}: {sz / 1e6:.0f}MB (total {got / 1e9:.1f}G)")

    sp = os.path.join(outdir, "fetch_stats.json")
    with open(sp, "w") as f:  # aggregate stats file; the durable record is the shards + log
        json.dump(stats, f, ensure_ascii=False, indent=1)
    print(f"{source}: {got / 1e9:.2f}G fetched -> {outdir}; stats {sp}")
    return 0  # partial target is fine: the harness records it, re-run resumes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="named source (fineweb2, cci3_hq, ...)")
    ap.add_argument("--target_bytes", type=float, default=None, help="disk bytes to fetch (None = all)")
    ap.add_argument("--stream_n", type=int, default=0, help="parallel streams (0 = one); fetch files where i%n==stream_i")
    ap.add_argument("--stream_i", type=int, default=0, help="this stream's index (0..stream_n-1)")
    a = ap.parse_args()
    return fetch(a.source, a.target_bytes or 0, a.stream_n, a.stream_i)


if __name__ == "__main__":
    sys.exit(main())
