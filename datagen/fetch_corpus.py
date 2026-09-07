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

    python datagen/fetch_corpus.py --source fineweb2 --target_bytes 30e9
    python datagen/fetch_corpus.py --source cci3_hq --target_bytes 50e9
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")


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


def _manifest_ms_starcoder_py():
    """code role, labelled Python (fb P0 ruling 2026-09-01: fetch a source that carries
    language labels so ast.parse filters syntax only, ~a few % loss, not language ID at
    94%). AI-ModelScope/starcoderdata python split, 59 parquet. ModelScope is the ONLY
    reachable host today (2026-09-01: hf-mirror + huggingface.co both 10s-timeout; MS
    served 206/0.75s on the probe). Flatten the local name to the basename so the .part
    path is writable (same fix as _manifest_ms_om2)."""
    names = open(os.path.join(ROOT, "data", "raw", "ms_starcoder_py_manifest.txt")).read().split()
    base = "https://www.modelscope.cn/datasets/AI-ModelScope/starcoderdata/resolve/master/"
    return [(n.split("/")[-1], base + n, 0) for n in names]


def _manifest_ms_finemath_4plus():
    """math cell (real source, ModelScope up while hf-mirror down 2026-08-31):
    AI-ModelScope/finemath finemath-4plus, 64 parquet (HuggingFaceTB/finemath's
    filtered math-web set). Resolve via modelscope.cn/datasets/<org>/<name>/resolve/master/."""
    names = open(os.path.join(ROOT, "data", "raw", "ms_finemath_4plus_manifest.txt")).read().split()
    base = "https://www.modelscope.cn/datasets/AI-ModelScope/finemath/resolve/master/"
    return [(n, base + n, 0) for n in names]



def _manifest_ms_om2():
    """cot role (ModelScope, 2026-08-31): OpenMathInstruct-2, 55 data parquet
    (small, under the finemath large-file abort; direct, no chunking). The
    manifest names carry a `data/` subdir prefix; the download writes
    `-o <out>/<name>.part`, so a non-empty dir part fails to create and curls
    exit 23 (write error) -- flatten the LOCAL name to the basename, keep the
    URL path so the file still resolves on the LFS repo."""
    names = open(os.path.join(ROOT, "data", "raw", "ms_om2_manifest.txt")).read().split()
    base = "https://www.modelscope.cn/datasets/AI-ModelScope/OpenMathInstruct-2/resolve/master/"
    return [(n.split("/")[-1], base + n, 0) for n in names]



def _manifest_hf_finemath_4plus():
    names = open(os.path.join(ROOT, "data", "raw", "hf_finemath_4plus_manifest.txt")).read().split()
    base = "https://hf-mirror.com/datasets/HuggingFaceTB/finemath/resolve/refs%2Fconvert%2Fparquet/finemath-4plus/train/"
    return [(n, base + n, 0) for n in names]


def _manifest_hf_om2():
    names = open(os.path.join(ROOT, "data", "raw", "ms_om2_manifest.txt")).read().split()
    base = "https://hf-mirror.com/datasets/open-math/OpenMathInstruct-2/resolve/main/"
    return [(n, base + n, 0) for n in names]



def _manifest_hf_numma():
    """cot (2026-08-31): AI-MO/NuminaMath-CoT via hf-mirror, 5 data parquet.
    Flat names (base points at data/) so the .part path is writable."""
    names = open(os.path.join(ROOT, "data", "raw", "hf_numma_manifest.txt")).read().split()
    base = "https://hf-mirror.com/datasets/AI-MO/NuminaMath-CoT/resolve/main/data/"
    return [(n, base + n, 0) for n in names]


def _manifest_hf_tree(dataset, data_dir="data"):
    """List parquet under HF <dataset>/<data_dir>/ via the tree API, pin the
    resolve base. Self-contained (no prebuilt manifest file); source_fp hashes
    the urls. curl -4: the pod's IPv6 egress is broken and urllib does not fall
    back (cot 2026-09-03, cot_criterion_0903)."""
    import subprocess
    import time
    api = f"https://hf-mirror.com/api/datasets/{dataset}/tree/main/{data_dir}"
    entries = None
    # The tree API flaps (a single curl can return an empty body while the mirror
    # is transiently down); the shard downloads survive via --retry but this
    # manifest read had none -- an empty response crashed the launch. Retry
    # before declaring failure (concurrent with the flapping, 2026-09-03).
    for attempt in range(4):
        out = subprocess.run(["curl", "-4", "-sL", "-m", "20", api],
                             capture_output=True, text=True)
        if out.stdout.strip():
            try:
                entries = json.loads(out.stdout)
                break
            except json.JSONDecodeError:
                pass
        time.sleep(3 * (attempt + 1))
    if entries is None:
        raise SystemExit(f"REFUSING: tree API for {dataset} returned no parseable JSON across 4 retries -- mirror down")
    files = [n for n in entries if n.get("type") == "file" and n.get("path", "").endswith(".parquet")]
    base = f"https://hf-mirror.com/datasets/{dataset}/resolve/main/{data_dir}/"
    return [(os.path.basename(n["path"]), base + os.path.basename(n["path"]), n.get("size") or 0)
            for n in files]


def _manifest_cot_open_thoughts():
    return _manifest_hf_tree("open-thoughts/OpenThoughts-114k")


def _manifest_cot_skywork_or1():
    return _manifest_hf_tree("Skywork/Skywork-OR1-RL-Data")


def _ot3_probe_ok(url):
    """True if the URL resolves to a final 200, following redirects (hf-mirror and
    modelscope both 302 to a CDN -- a HEAD probing the FIRST response line calls
    an available mirror 'down'). curl -4 (pod IPv6 egress broken), 10 s timeout
    (the mirror-rule floor), final status judged after -L. 2026-09-04: hf-mirror
    302->AWS CDN, modelscope 302->cdn-lfs-cn, both final 200; the pre-fix -sI
    rejected the 302 and REFUSEd a fetchable source."""
    import subprocess
    p = subprocess.run(["curl", "-4", "-sIL", "-o", "/dev/null", "-w", "%{http_code}", "-m", "10", url],
                       capture_output=True, text=True, timeout=12)
    return p.returncode == 0 and p.stdout.strip() == "200"


def _manifest_cot_ot3():
    """32 of 120 slices for the cot_ot3 role (aupai-6e 2026-09-03): depends on
    NO tree API -- the filenames follow the pattern train-000NN-of-00120.parquet,
    so the list is built from the pattern, then each resolve path is probed
    (hf-mirror first, modelscope second) and the one that answers a final 200
    (redirects followed) is used. fetch_stats records which host served each file."""
    base_hf = "https://hf-mirror.com/datasets/open-thoughts/OpenThoughts3-1.2M/resolve/main/data/"
    base_ms = "https://www.modelscope.cn/api/v1/datasets/open-thoughts/OpenThoughts3-1.2M/repo?FilePath=data/"
    out = []
    for i in range(32):
        name = f"train-{i:05d}-of-00120.parquet"
        for base, host in ((base_hf, "hf-mirror.com"), (base_ms, "www.modelscope.cn")):
            url = f"{base}{name}"
            if _ot3_probe_ok(url):
                out.append((name, url, 0))
                break
        else:
            print(f"  {name}: no host answered a final 200 (hf-mirror/modelscope) -- skipped", file=__import__("sys").stderr, flush=True)
    if not out:
        raise SystemExit("REFUSING: none of the 32 OT3 slices resolved on hf-mirror or modelscope")
    return out


def _manifest_cot_openr1():
    return _manifest_hf_tree("open-r1/OpenR1-Math-220k")


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
    "hf_numma": _manifest_hf_numma,
    "cot_open_thoughts": _manifest_cot_open_thoughts,
    "cot_skywork_or1": _manifest_cot_skywork_or1,
    "cot_ot3": _manifest_cot_ot3,
    "cot_openr1": _manifest_cot_openr1,
    "ms_starcoder_py": _manifest_ms_starcoder_py,
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


def _mirror_chain(url, modelscope_url=None):
    """Per-source mirror chain, most-to-least-preferred, same relative path.
    Default: hf-mirror then huggingface.co (a straightforward host swap, since a
    failing mirror at fetch time is the case that lost a day, 2026-08-31/30).
    ModelScope is the SECOND host for sources that specify it (its name map is
    not a 1:1 host swap, so it is source-supplied) -- and it gets the NO-resume
    `-C -` skip, because ModelScope's LFS aborts on a range request."""
    chain = [url]
    if url.startswith("https://hf-mirror.com/"):
        chain.append(url.replace("https://hf-mirror.com/", "https://huggingface.co/", 1))
    if modelscope_url:
        chain.append(modelscope_url)
    return chain


def _ranged_get(url, part, name, chunks=8, **chunk_opts):
    """Download one file as `chunks` parallel byte ranges. (True, bytes) or (False, why).

    The default since 2026-09-07 (§259): data.together.xyz throttles a LONG-LIVED
    connection to ~40 KB/s while answering a fresh one at 7.1 MB/s -- measured 177x
    apart on the same host in the same second, and confirmed by one fetch whose second
    file reached 651 MB while its first sat at 29 MB. Nothing errors, nothing retries,
    the .part grows monotonically; the rate is the only symptom, so a probe taken at the
    start of a transfer measures the fast case by construction and says nothing about
    the regime a minute later. 20.2 GB went from 137 h to 0.7 h.

    The assembled size is checked against content-length because concatenating N chunk
    files turns a silently short chunk into a complete-looking download, and a truncated
    jsonl still parses for most of its rows -- the failure would land in the corpus, not
    in this function. A server that ignores Range (200 instead of 206) or hides its length
    gets (False, why) and the caller falls back to the single stream.
    """
    head = subprocess.run(["curl", "-4", "-sI", "-m", "30", url], capture_output=True, text=True)
    size = None
    accepts_ranges = False
    for line in head.stdout.splitlines():
        low = line.lower()
        if low.startswith("content-length:"):
            try:
                size = int(line.split(":", 1)[1].strip())
            except ValueError:
                size = None
        elif low.startswith("accept-ranges:") and "bytes" in low:
            accepts_ranges = True
    if not size:
        return False, "no content-length"
    if not accepts_ranges:
        return False, "no accept-ranges: bytes"
    if size < 64 * 1024 * 1024:
        return False, f"{size}B below the 64MB chunking floor"

    per = size // chunks
    spans = []
    for i in range(chunks):
        lo = i * per
        hi = size - 1 if i == chunks - 1 else lo + per - 1
        spans.append((lo, hi, f"{part}.c{i}"))
    ok, why = _run_chunks(url, spans, name, **chunk_opts)
    if not ok:
        # Only the chunks that never completed are removed. The sibling chunks stay:
        # deleting them is what cost 1.8 GB of a 2.02 GB file on 2026-09-07, when killing
        # two throttled connections took the whole-file failure branch and it swept the six
        # that had finished. A partial download is worth more than a clean directory.
        return False, why
    with open(part, "wb") as out:
        for _lo, _hi, cp in spans:
            with open(cp, "rb") as f:
                shutil.copyfileobj(f, out)
            _rm(cp)
    got = os.path.getsize(part)
    if got != size:
        _rm(part)
        return False, f"assembled {got}B, content-length {size}B"
    print(f"  {name}: {chunks}-way ranged, {got}B verified", file=sys.stderr, flush=True)
    return True, got


CHUNK_FLOOR_BPS = 500 * 1024   # a chunk under this over one window is throttled, not slow
CHUNK_WINDOW_S = 60            # long enough that a brief stall is not a restart
CHUNK_MAX_RESTARTS = 5         # per chunk; past this the host is the problem, not the socket


def _run_chunks(url, spans, name, floor=CHUNK_FLOOR_BPS, window=CHUNK_WINDOW_S,
                max_restarts=CHUNK_MAX_RESTARTS, _sleep=time.sleep, _now=time.monotonic):
    """Download every span, restarting a chunk whose rate falls under `floor`.

    Parallelism buys the head, restart buys the tail (4c, 2026-09-07). Measured on
    data.together.xyz: 8 chunks of one 2.02 GB file, six finished at 252,793,470 B while
    two sat throttled holding 218 MB between them -- the file finishes when its SLOWEST
    chunk does, so chunking alone converts a bandwidth problem into a tail-latency one.
    A throttled connection stays throttled for its whole life and a fresh one is instantly
    fast (7.1 MB/s against 40 KB/s, same host, same second), so the fix is to notice and
    reconnect rather than wait.

    A restart RESUMES: the bytes already on disk are kept and the range is re-issued from
    where it stopped. Restarting from zero would discard exactly what the throttled chunk
    did manage, and the slow chunks are the ones with the most to lose.

    The same applies ACROSS calls. A span whose base file already holds its full length is
    not fetched again, and a partial base is resumed -- so a run that failed on one chunk
    costs one chunk, not the file. Without this, "the siblings are not deleted" would be
    an empty promise: the next attempt would truncate and refetch them anyway.
    """
    live = {}   # i -> (Popen, bytes_at_window_start, window_start)
    restarts = [0] * len(spans)
    done, failed = set(), {}

    def _launch(i, resume_from=None):
        lo, hi, cp = spans[i]
        start = lo if resume_from is None else lo + resume_from
        p = subprocess.Popen(
            ["curl", "-4", "-sS", "--fail", "--retry", "4", "--retry-delay", "5",
             "-r", f"{start}-{hi}", "-o", cp if resume_from is None else f"{cp}.r{restarts[i]}", url],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        live[i] = [p, _chunk_bytes(spans[i], restarts[i]), _now()]

    for i in range(len(spans)):
        lo, hi, cp = spans[i]
        want = hi - lo + 1
        _consolidate(spans[i], 0)
        have = _chunk_bytes(spans[i], 0)
        if have == want:
            done.add(i)
            continue
        if have > want:
            _rm(cp)
            have = 0
        if have:
            restarts[i] = 1
            _launch(i, resume_from=have)
        else:
            _launch(i)

    while live:
        _sleep(min(5, window))
        for i in list(live):
            p, mark, t0 = live[i]
            rc = p.poll()
            if rc is not None:
                if rc == 0:
                    done.add(i)
                else:
                    failed[i] = f"rc {rc}"
                del live[i]
                continue
            elapsed = _now() - t0
            if elapsed < window:
                continue
            now_bytes = _chunk_bytes(spans[i], restarts[i])
            rate = (now_bytes - mark) / elapsed
            if rate >= floor:
                live[i] = [p, now_bytes, _now()]
                continue
            if restarts[i] >= max_restarts:
                print(f"  {name}: chunk {i} still {rate / 1024:.0f} KB/s after "
                      f"{max_restarts} restarts; letting it run", file=sys.stderr, flush=True)
                live[i] = [p, now_bytes, _now()]
                continue
            p.kill()
            p.wait()
            _consolidate(spans[i], restarts[i])
            restarts[i] += 1
            have = _chunk_bytes(spans[i], restarts[i])
            print(f"  {name}: chunk {i} at {rate / 1024:.0f} KB/s -> restart "
                  f"{restarts[i]} from {have}B", file=sys.stderr, flush=True)
            _launch(i, resume_from=have)

    for i in sorted(done):
        _consolidate(spans[i], restarts[i])
    if failed:
        for i in failed:
            for suffix in range(restarts[i] + 1):
                _rm(f"{spans[i][2]}.r{suffix}")
        return False, f"chunk(s) {sorted(failed)} failed: {failed}"
    return True, None


def _chunk_bytes(span, restarts):
    """Bytes on disk for one span: the base file plus every restart fragment."""
    _lo, _hi, cp = span
    total = os.path.getsize(cp) if os.path.exists(cp) else 0
    for r in range(restarts + 1):
        f = f"{cp}.r{r}"
        if os.path.exists(f):
            total += os.path.getsize(f)
    return total


def _consolidate(span, restarts):
    """Fold a chunk's restart fragments back into its base file, in order."""
    _lo, _hi, cp = span
    for r in range(restarts + 1):
        f = f"{cp}.r{r}"
        if not os.path.exists(f):
            continue
        with open(cp, "ab") as base, open(f, "rb") as frag:
            shutil.copyfileobj(frag, base)
        _rm(f)


def _rm(p):
    try:
        os.remove(p)
    except OSError:
        pass


def _fetch_one(url_chain, part, name, prev_host):
    """Probe + download one shard across the host chain; continue the SAME .part
    on the next host. Returns (subprocess.CompletedProcess|None, serving_host).
    A host that does not answer a 10 s IPv4 HEAD is abandoned in ~10 s and the
    next serves; ModelScope (the no-resume LFS) is downloaded whole, no -C -.

    Each host is tried with an 8-way ranged download FIRST (§259) and falls back to the
    single stream when the server will not serve ranges or the assembly does not verify.
    The fallback is not a formality: ModelScope's LFS aborts on a range request, which is
    the same reason `-C -` is already skipped for it below."""
    server = None
    for u in url_chain:
        probe = subprocess.run(
            ["curl", "-4", "-sI", "-m", "10", u],
            capture_output=True, text=True, timeout=12,
        )
        if probe.returncode != 0 or not probe.stdout.startswith(("HTTP/", "HTTP/")):
            print(f"  {name}: host {_host(u)} unreachable (rc {probe.returncode}) -> next", file=sys.stderr, flush=True)
            continue
        server = _host(u)
        if "modelscope" not in u and not os.path.exists(part):
            ok, why = _ranged_get(u, part, name)
            if ok:
                return subprocess.CompletedProcess([], 0), server
            print(f"  {name}: ranged fetch unavailable ({why}) -> single stream", file=sys.stderr, flush=True)
        args = ["curl", "-4", "-sL", "-o", part, "--retry", "6", "--retry-delay", "3", u]
        if "modelscope" not in u:
            args[3:3] = ["-C", "-"]  # resume only off ModelScope (its LFS aborts on range)
        r = subprocess.run(args, stdout=subprocess.DEVNULL)
        if r.returncode == 0:
            return r, server
        print(f"  {name}: {_host(u)} download rc {r.returncode} -> next", file=sys.stderr, flush=True)
    return subprocess.CompletedProcess([], 1), server


def _host(u):
    return u.split("/")[2] if u.startswith("http") else u


def _chain_hosts(url_chain):
    """Hosts of a mirror chain, for the give-up message. t37 reading: a source
    missing on ALL its hosts must name the chain it tried, so the operator sees
    every mirror probed, not just the last."""
    return ", ".join(_host(u) for u in url_chain)


def _selftest():
    """t37 acceptance, hermetic (no internet): a local HTTP server serves one
    shard; the FIRST mirror host is a closed port. Assert the closed host is
    abandoned in <10 s and the server host serves the bytes with the serving
    host recorded; assert an all-closed chain fails and names every host."""
    import http.server
    import tempfile
    import threading

    payload = b"hello t37 mirror chain\n" * 3
    # 80MB, over _ranged_get's 64MB floor. Content varies by offset so a chunk assembled in
    # the wrong order, or a chunk served from the wrong range, does not compare equal.
    big_payload = bytes((i * 7 + (i >> 13)) & 0xFF for i in range(80 * 1024 * 1024))
    # The slow/bad worlds address chunk 3 of 8 by its span start, so the assertions name a
    # specific chunk rather than "one of them".
    CH = (80 * 1024 * 1024) // 8
    SLOW_LO = 3 * CH
    BAD_LO = 3 * CH
    SLOW_PREFIX = 4096          # what the throttled range delivers before it stops
    STALL_S = 30                # >> the selftest's 1 s window, so the floor must trip
    ranges_seen = []            # every range start the server was asked for

    class H(http.server.BaseHTTPRequestHandler):
        def _ok(self):
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Connection", "close")
            self.end_headers()

        def _ok(self):
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Connection", "close")
            self.end_headers()

        def _route(self, body):
            if self.path == "/red":
                self.send_response(302)
                self.send_header("Location", "/x.jsonl")
                self.send_header("Connection", "close")
                self.end_headers()
            elif self.path == "/missing":
                self.send_error(404)
            elif self.path in ("/big.jsonl", "/short.jsonl", "/norange.jsonl",
                               "/slow.jsonl", "/slowcap.jsonl", "/onebad.jsonl"):
                self._big(body)
            elif self.path == "/small.jsonl":
                # range-capable but tiny: the only route that reaches the size floor
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Connection", "close")
                self.end_headers()
                if body:
                    self.wfile.write(payload)
            else:
                self._ok()
                if body:
                    self.wfile.write(payload)

        def _big(self, body):
            """The ranged-path worlds. /big serves ranges honestly; /short declares the
            full length and returns one byte less per range, the corruption the size check
            exists for; /norange omits Accept-Ranges so the caller must fall back; /slow and
            /slowcap stall ONE range mid-body (a throttled connection, not an error);
            /onebad 404s one range while the others succeed."""
            n = len(big_payload)
            rng = self.headers.get("Range")
            if self.path == "/norange":
                rng = None
            if rng and self.path != "/norange.jsonl":
                lo, hi = rng.split("=", 1)[1].split("-")
                lo, hi = int(lo), int(hi or n - 1)
                ranges_seen.append(lo)
                if self.path == "/onebad.jsonl" and lo == BAD_LO:
                    self.send_error(404)
                    return
                chunk = big_payload[lo:hi + 1]
                if self.path == "/short.jsonl":
                    chunk = chunk[:-1]  # one byte short per chunk: assembly must not verify
                # The throttle, reproduced: a prefix arrives, then the connection delivers
                # nothing for STALL_S. Only the ORIGINAL offset stalls, so a resumed range
                # (lo + bytes already on disk) is served at full speed -- which is the
                # measured behaviour the restart exists to exploit.
                stall = lo == SLOW_LO and self.path in ("/slow.jsonl", "/slowcap.jsonl")
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {lo}-{hi}/{n}")
                self.send_header("Content-Length", str(len(chunk)))
                self.send_header("Connection", "close")
                self.end_headers()
                if body:
                    if stall:
                        self.wfile.write(chunk[:SLOW_PREFIX])
                        self.wfile.flush()
                        time.sleep(STALL_S)
                        try:
                            self.wfile.write(chunk[SLOW_PREFIX:])
                        except BrokenPipeError:
                            # the restart killed this connection, which is the point of
                            # the test; without the catch the server thread prints a
                            # traceback and a green run reads as a failure
                            pass
                    else:
                        self.wfile.write(chunk)
                return
            self.send_response(200)
            self.send_header("Content-Length", str(n))
            if self.path != "/norange.jsonl":
                self.send_header("Accept-Ranges", "bytes")
            self.send_header("Connection", "close")
            self.end_headers()
            if body:
                self.wfile.write(big_payload)

        def do_HEAD(self):
            self._route(body=False)  # probe uses curl -I; the 302/404 must answer on HEAD

        def do_GET(self):
            self._route(body=True)

        def log_message(self, *a):
            pass

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    d = tempfile.mkdtemp()
    part = os.path.join(d, "shard.part")
    good = f"http://127.0.0.1:{port}/x.jsonl"
    red = f"http://127.0.0.1:{port}/red"      # 302 -> /x.jsonl: fetchable behind a redirect
    missing = f"http://127.0.0.1:{port}/missing"  # 404: must be judged down
    closed_a = "http://127.0.0.1:9/x.jsonl"  # discard port: refused, fails fast
    closed_b = "http://127.0.0.1:8/y.jsonl"
    try:
        # (a1) the OT3-style redirect probe judges the FINAL status, not the 302 first line
        assert _ot3_probe_ok(red) is True, f"302->200 probe must be True, got {_ot3_probe_ok(red)}"
        assert _ot3_probe_ok(missing) is False, "404 probe must be False"
        assert _ot3_probe_ok(closed_a) is False, "closed port probe must be False"
        # (a) failover: closed first host abandoned, server host serves the bytes
        r, server = _fetch_one([closed_a, good], part, "t37selftest", None)
        assert r.returncode == 0, f"failover did not serve: rc {r.returncode}"
        with open(part, "rb") as fp:
            got_bytes = fp.read()
        assert got_bytes == payload, "served bytes mismatch"
        assert server == f"127.0.0.1:{port}", f"wrong serving host {server}"
        # (b) missing on ALL hosts: fails, names the chain tried
        r2, server2 = _fetch_one([closed_a, closed_b], part, "t37selftest", None)
        assert r2.returncode != 0, "all-closed chain must fail"
        assert server2 is None, f"no host should serve an all-closed chain, got {server2}"
        got = _chain_hosts([closed_a, closed_b])
        assert "127.0.0.1:9" in got and "127.0.0.1:8" in got, f"chain not named: {got!r}"

        # (c) the ranged path (§259). Its decision logic is what a wrong answer costs a
        # corpus, so each refusal reason is asserted separately rather than "it fell back".
        big_part = os.path.join(d, "big.part")
        ok, why = _ranged_get(f"http://127.0.0.1:{port}/big.jsonl", big_part, "rangetest")
        assert ok, f"a range-serving host with a large body must use the ranged path: {why}"
        with open(big_part, "rb") as fp:
            assert fp.read() == big_payload, "ranged assembly did not reproduce the body"
        # THE ASSERTION THAT MATTERS: a server that lies about length must NOT yield a file.
        # Concatenating N chunks turns a short chunk into a complete-looking download, and a
        # truncated jsonl parses for most of its rows, so this failure would land in the
        # corpus rather than here.
        _rm(big_part)
        ok2, why2 = _ranged_get(f"http://127.0.0.1:{port}/short.jsonl", big_part, "shorttest")
        assert not ok2, "a body shorter than its content-length must be refused"
        assert "assembled" in why2, f"the refusal must name the size mismatch, got {why2!r}"
        assert not os.path.exists(big_part), "a mismatched assembly must leave no file behind"
        # no Accept-Ranges -> fall back rather than issue ranges the server ignores
        ok3, why3 = _ranged_get(f"http://127.0.0.1:{port}/norange.jsonl", big_part, "norangetest")
        assert not ok3 and "accept-ranges" in why3, f"no-range host must fall back: {why3!r}"
        # below the floor -> single stream; chunking a small file costs 8 connections for
        # nothing. Served through /small.jsonl, which DOES advertise ranges: the plain /x.jsonl
        # route sends no Accept-Ranges, so it refuses one step earlier and would have made this
        # assertion pass without ever reaching the floor -- a test green for the wrong reason.
        ok4, why4 = _ranged_get(f"http://127.0.0.1:{port}/small.jsonl", big_part, "smalltest")
        assert not ok4 and "floor" in why4, f"a small body must skip chunking: {why4!r}"

        # (d) restart-on-slow-chunk. One range stalls after 4096 B for 30 s while its seven
        # siblings finish; the floor must notice, kill THAT connection, re-issue the rest of
        # its range, and assemble a byte-exact file. Window 1 s and floor 1 MB/s so the world
        # runs in seconds; the production constants (60 s / 500 KB/s) are the same code path.
        _rm(big_part)
        slow_opts = dict(floor=1024 * 1024, window=1.0)
        ok5, why5 = _ranged_get(f"http://127.0.0.1:{port}/slow.jsonl", big_part, "slowtest",
                                **slow_opts)
        assert ok5, f"a stalled chunk must be restarted, not waited out: {why5}"
        with open(big_part, "rb") as fp:
            assert fp.read() == big_payload, "a restarted chunk must reassemble byte-exact"
        # The restart RESUMES: the stalled range delivered 4096 B before stalling, and those
        # bytes are in the output above. A restart-from-zero would also reassemble correctly,
        # so the byte check alone cannot tell the two apart -- assert the resumed request was
        # actually issued at the resumed offset.
        assert any(o >= SLOW_LO + SLOW_PREFIX for o in ranges_seen), (
            f"no range was re-issued past the stall point {SLOW_LO + SLOW_PREFIX}; "
            f"offsets seen: {sorted(set(ranges_seen))}")

        # (e) the cap: with max_restarts=0 the stall is never restarted, so the same world
        # must instead take ~STALL_S and still succeed. This is the negative control for (d)
        # -- without it, a floor that never fires would pass (d) too, since the stalled chunk
        # eventually completes on its own.
        _rm(big_part)
        t_cap = time.monotonic()
        ok6, why6 = _ranged_get(f"http://127.0.0.1:{port}/slowcap.jsonl", big_part, "slowcaptest",
                                floor=1024 * 1024, window=1.0, max_restarts=0)
        cap_s = time.monotonic() - t_cap
        assert ok6, f"a capped-out chunk must be left to finish, not failed: {why6}"
        assert cap_s >= STALL_S * 0.8, (
            f"max_restarts=0 waited {cap_s:.1f}s, less than the {STALL_S}s stall -- the stall "
            f"world is not stalling, so (d) proved nothing")

        # (f) a failed chunk must NOT delete its completed siblings. This is the 2026-09-07
        # incident as a test: one range 404s, seven finish, and the seven stay on disk.
        _rm(big_part)
        ok7, why7 = _ranged_get(f"http://127.0.0.1:{port}/onebad.jsonl", big_part, "onebadtest")
        assert not ok7, "a 404 range must fail the file"
        survivors = [i for i in range(8) if os.path.exists(f"{big_part}.c{i}")]
        assert len(survivors) == 7, (
            f"a failed chunk swept its siblings: {len(survivors)} of 8 survived ({survivors})")
        # and the next call resumes from them rather than refetching: it must succeed against
        # the honest route while the seven survivors are already on disk.
        ok8, why8 = _ranged_get(f"http://127.0.0.1:{port}/big.jsonl", big_part, "resumetest")
        assert ok8, f"a resume over surviving chunks must complete: {why8}"
        with open(big_part, "rb") as fp:
            assert fp.read() == big_payload, "resume over survivors did not reproduce the body"

        print(f"fetch_corpus selftest OK: failover served {len(payload)}B on {server}; "
              f"all-closed named {got!r}; ranged path verified {len(big_payload)}B in 8 chunks "
              f"and refused a short body, a no-range host and a sub-floor body; a stalled chunk "
              f"was restarted from its resumed offset, the cap waited {cap_s:.0f}s instead, and "
              f"a 404 chunk left 7 of 8 siblings on disk for the resume")
        return 0
    finally:
        httpd.shutdown()
        shutil.rmtree(d, ignore_errors=True)


def fetch(source, target_bytes, stream_n=0, stream_i=0, modelscope_urls=None):
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
        # backoff, and -- when the (optional) mirror chain is set -- fail over to
        # the next host, continuing the SAME .part, recording the serving host.
        attempts = 0
        serving = {}
        url_chain = _mirror_chain(url)
        while True:
            r, host = _fetch_one(url_chain, part, name, serving.get("host"))
            if r.returncode == 0:
                serving["host"] = host
                break
            attempts += 1
            print(
                f"  {name}: curl exit {r.returncode} (stream/net err) on {host or url_chain[0]}; "
                f"attempt {attempts}; resuming .part, backing off",
                file=sys.stderr,
                flush=True,
            )
            if attempts >= 4:
                print(
                    f"  {name}: giving up this shard after {attempts} outer retries; "
                    f"hosts tried ({_chain_hosts(url_chain)}) all failed; "
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
        rec = {"shard": name, "bytes": sz, "status": "fetched", "host": serving.get("host") or _host(url)}
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
    ap.add_argument("--selftest", action="store_true", help="run the hermetic mirror-chain selftest and exit")
    ap.add_argument("--source", default=None, help="named source (fineweb2, cci3_hq, ...)")
    ap.add_argument("--target_bytes", type=float, default=None, help="disk bytes to fetch (None = all)")
    ap.add_argument("--stream_n", type=int, default=0, help="parallel streams (0 = one); fetch files where i%%n==stream_i")
    ap.add_argument("--stream_i", type=int, default=0, help="this stream's index (0..stream_n-1)")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    if not a.source:
        ap.error("--source <name> is required (or pass --selftest)")
    return fetch(a.source, a.target_bytes or 0, a.stream_n, a.stream_i)


if __name__ == "__main__":
    sys.exit(main())
