#!/usr/bin/env python3
"""Build the pretraining corpus: select, clean, dedup, cap — the source is interchangeable.

    python datagen/build_corpus.py --domain web  --source fineweb2 --target_tokens 6e9
    python datagen/build_corpus.py --domain math --source jsonl:data/synthetic/math_*.jsonl --target_tokens 1e9
    python datagen/build_corpus.py --source jsonl:data/raw/*.jsonl --dry --limit 2000   # inspect rejects

Output: data/corpus/<domain>/<source>_NNN.jsonl (100MB shards, {"content","source","url"}), plus a
rejects histogram. Filters run in order of cost; see reject_reason().
"""

import argparse
import glob
import gzip
import hashlib
import json
import os
import re
import struct
import sys
from collections import Counter
from urllib.parse import urlsplit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from holdout import is_holdout  # noqa: E402
from loader import format_example  # noqa: E402

OUT_DIR = os.path.join(ROOT, "data", "corpus")
SHARD_BYTES = 100 * 2**20
CHARS_PER_TOKEN = 1.5
REJECT_EARLY_AT = 20_000  # fast-fail: >95% single-reason reject by this many docs -> wrong filters

SOURCES = {
    "fineweb2": ("HuggingFaceFW/fineweb-2", "data/cmn_Hani/train/", "text", "url"),
    "skypile": ("Skywork/SkyPile-150B", "data/", "text", None),
}

# Control tokens leak in from distilled corpora; strip them or the "unfinished"
# rule rejects the documents.
SPECIAL_TOKEN = re.compile(r"<\|[A-Za-z0-9_]+\|>")
CJK = re.compile(r"[一-鿿㐀-䶿]")
BAD = re.compile(r"[�\x00-\x08\x0b\x0c\x0e-\x1f]")
SYMBOL = re.compile(r"[\d\W]", re.UNICODE)
URL = re.compile(r"https?://|www\.|[\w.-]+@[\w-]+\.\w+")
END_OK = re.compile(r"[。！？!?…”」』）)》\]】]\s*$|[。！？]\s*[\"'”」]?\s*$")
BOILER = re.compile(
    r"版权所有|ICP备|点击进入|免责声明|更多精彩|关注公众号|微信号|扫码|下载APP|本文来源|转载请|阅读全文"
    r"|上一篇|下一篇|网友评论|登录后|立即注册|广告|友情链接|联系我们|站长统计|Copyright|All Rights Reserved"
)


def load_garbage_patterns():
    # A missing pattern file must raise, not silently disable the filter.
    # AUPAI_NO_GARBAGE=1 disables explicitly: garbage_topic false-positives on
    # para-athletes and dinosaurs.
    if os.environ.get("AUPAI_NO_GARBAGE") == "1":
        return None
    pats = []
    for name in ("pass1_garbage", "pass2_garbage", "pass3_garbage"):
        path = os.path.join(ROOT, "filters", f"{name}.py")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} missing; set AUPAI_NO_GARBAGE=1 to run without it")
        ns = {}
        exec(compile(open(path, encoding="utf-8").read(), path, "exec"), ns)
        pats += ns.get("PATTERNS", [])
    assert pats, "garbage pattern files loaded but PATTERNS is empty"
    return re.compile("|".join(f"(?:{p})" for p in pats))


GARBAGE = load_garbage_patterns()


# Holdout probes must catch the role marker, multi-line and <15-char questions a
# per-line hash misses. QA_PREFIX keeps pre-ChatML 问：/答： forms: the corpus
# still holds documents written that way.
QA_PREFIX = re.compile(
    r"^\s*(?:<\|im_start\|>(?:user|assistant|system)\s*|问题?|答案?|Q|A|Question|Answer)\s*[：:]?\s*"
)
ANSWER_TAIL = re.compile(r"(?:<\|im_end\|>|\n\s*(?:答案?|A|Answer)\s*[：:])")


def reject_holdout(text):
    if is_holdout(text):
        return "eval_contaminated"
    body = QA_PREFIX.sub("", ANSWER_TAIL.split(text, 1)[0]).strip()
    if body != text and is_holdout(body):
        return "eval_contaminated"
    for ln in (ln.strip() for ln in text.split("\n")):
        for cand in {ln, QA_PREFIX.sub("", ln)}:
            if cand and len(cand) <= 500 and is_holdout(cand):
                return "eval_contaminated"
    return None


def reject_light(text):
    """Domain-neutral checks only: the web chain's CJK/symbol/digit ratios delete a code or
    English corpus outright."""
    n = len(text)
    if n < 100:
        return "short"
    if n > 200_000:
        return "long"
    if len(BAD.findall(text)) > 0.001 * n:
        return "bad_bytes"
    return reject_holdout(text)


def reject_reason(text):
    """None if the document is kept, else a short reason (for the histogram)."""
    n = len(text)
    if n < 200:
        return "short"
    if n > 50_000:
        return "long"
    nonspace = sum(1 for c in text if not c.isspace()) or 1
    if len(CJK.findall(text)) / nonspace < 0.6:
        return "not_zh"
    if len(BAD.findall(text)) > 0.001 * n:
        return "bad_bytes"
    if len(SYMBOL.findall(text)) / nonspace > 0.35:
        return "symbols"
    if sum(c.isdigit() for c in text) / nonspace > 0.2:
        return "digits"
    if len(BOILER.findall(text)) >= 3:
        return "boilerplate"
    if len(URL.findall(text)) > 3 * n / 1000:
        return "urls"
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if len(lines) >= 5:
        if sum(len(ln) < 10 for ln in lines) / len(lines) > 0.5:
            return "nav_menu"
        if len(set(lines)) / len(lines) < 0.7:
            return "dup_lines"
    # 500, not 2000: above 500 chars an "unfinished" tail is a source credit or
    # tag list on a complete article.
    if n < 500 and not END_OK.search(text):
        return "unfinished"
    if GARBAGE is not None and GARBAGE.search(text[:600]):
        return "garbage_topic"
    return reject_holdout(text)


# ---- dedup -------------------------------------------------------------------------------------
_NORM = re.compile(r"[\s\W_]+", re.UNICODE)


def exact_key(text):
    return hashlib.sha1(_NORM.sub("", text).encode("utf-8")).digest()[:12]


class MinHashLSH:
    """128-perm MinHash over char 5-gram shingles, 16 bands x 8 rows (~0.8 Jaccard threshold).
    ponytail: pure-python, in-memory; fine to ~30M docs."""

    def __init__(self, perms=128, bands=16, seed=17):
        import random

        rng = random.Random(seed)
        self.mask = (1 << 61) - 1
        self.ab = [(rng.randrange(1, self.mask), rng.randrange(0, self.mask)) for _ in range(perms)]
        self.bands, self.rows = bands, perms // bands
        self.tables = [dict() for _ in range(bands)]

    def signature(self, text):
        s = _NORM.sub("", text)
        shingles = {s[i : i + 5] for i in range(max(1, len(s) - 4))}
        hs = [
            int.from_bytes(hashlib.blake2b(sh.encode(), digest_size=8).digest(), "little") for sh in shingles
        ]
        sig = []
        for a, b in self.ab:
            sig.append(min(((a * h + b) & self.mask) for h in hs))
        return sig

    def seen(self, text):
        """True if a near-duplicate was inserted before; inserts otherwise."""
        sig = self.signature(text)
        keys = [
            struct.pack(f"{self.rows}Q", *sig[i * self.rows : (i + 1) * self.rows]) for i in range(self.bands)
        ]
        if any(k in t for k, t in zip(keys, self.tables, strict=True)):
            return True
        for k, t in zip(keys, self.tables, strict=True):
            t[k] = 1
        return False


# ---- sources -----------------------------------------------------------------------------------
def iter_jsonl(path):
    op = gzip.open if path.endswith(".gz") else open
    bad = 0
    with op(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    # One malformed line killed the whole en_c4 clean (2026-08-31):
                    # a worker raised, pool.map re-raised, the run died at 90%.
                    bad += 1
                    continue
                text = d.get("content") or d.get("text")
                if not text and d.get("instruction"):
                    # Same ChatML the SFT pack uses, so pretraining already sees that format.
                    text = "".join(format_example(d["instruction"], d.get("output", "")))
                yield text or "", d.get("url")
    if bad:
        print(f"iter_jsonl: {bad} malformed line(s) skipped in {path}", file=sys.stderr)


def iter_parquet(path, text_col="text", url_col="url", rg_mod=None, rg_idx=None):
    """Yield (text, url) from a parquet file. rg_mod/rg_idx select a shard's row
    groups (i % rg_mod == rg_idx) so N parallel processes can read one file without
    rewriting it -- build_corpus is single-threaded, and per-file parallelism is
    the way to use all cores on one downloaded parquet."""
    import pyarrow.parquet as pq

    f = pq.ParquetFile(path)
    cols = [c for c in (text_col, url_col) if c in f.schema_arrow.names]
    groups = (
        range(f.num_row_groups)
        if not rg_mod
        else [i for i in range(f.num_row_groups) if i % rg_mod == rg_idx]
    )
    for g in groups:
        d = f.read_row_group(g, columns=cols).to_pydict()
        for i in range(len(d[text_col])):
            yield d[text_col][i] or "", (d[url_col][i] if url_col in d else None)


def iter_source(spec, cache_dir, rg_mod=None, rg_idx=None):
    """Yield (text, url, source_name)."""
    if spec.startswith("parquet:"):
        for p in sorted(glob.glob(spec[8:])):
            name = os.path.basename(p).split(".")[0]
            for text, url in iter_parquet(p, rg_mod=rg_mod, rg_idx=rg_idx):
                yield text, url, name
        return
    if spec.startswith("jsonl:"):
        for p in sorted(glob.glob(spec[6:])):
            for text, url in iter_jsonl(p):
                yield text, url, os.path.basename(p).split(".")[0]
        return
    repo, prefix, text_col, url_col = SOURCES[spec]
    from huggingface_hub import hf_hub_download, list_repo_files

    files = sorted(f for f in list_repo_files(repo, repo_type="dataset") if f.startswith(prefix))
    for f in files:
        local = hf_hub_download(repo, f, repo_type="dataset", local_dir=cache_dir)
        if f.endswith(".parquet"):
            import pyarrow.parquet as pq

            cols = [text_col] + ([url_col] if url_col else [])
            for batch in pq.ParquetFile(local).iter_batches(batch_size=4096, columns=cols):
                d = batch.to_pydict()
                for i in range(len(d[text_col])):
                    yield d[text_col][i] or "", (d[url_col][i] if url_col else None), spec
        else:
            for text, url in iter_jsonl(local):
                yield text, url, spec
        try:
            os.remove(local)  # one shard on disk at a time
        except OSError:
            pass


# ---- main --------------------------------------------------------------------------------------
class ShardWriter:
    def __init__(self, out_dir, prefix):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir, self.prefix, self.n, self.f, self.size = out_dir, prefix, 0, None, 0

    def _path(self, part):
        p = os.path.join(self.out_dir, f"{self.prefix}_{self.n:03d}.jsonl")
        return p + ".part" if part else p

    def write(self, row):
        if self.f is None or self.size >= SHARD_BYTES:
            if self.f:
                self.f.close()
                # rename the shard JUST closed (self.n still its index), then
                # advance -- renaming after advancing looked for the wrong path
                # (e2e282f bug: closed _000, renamed _001 => FileNotFoundError).
                os.rename(self._path(True), self._path(False))
                self.n += 1
            self.f = open(self._path(True), "w", encoding="utf-8")
            self.size = 0
        line = json.dumps(row, ensure_ascii=False) + "\n"
        self.f.write(line)
        self.size += len(line.encode("utf-8"))

    def close(self):
        if self.f:
            self.f.close()
            if os.path.exists(self._path(True)):
                os.rename(self._path(True), self._path(False))


def _clean_piece(piece):
    """One worker's slice: (file specs, filters name, out dir, shard prefix,
    rgmod, rgidx, cache_dir, dry). Per-worker exact dedup + shards prefixed
    with the worker id so no two workers write the same file. Returns
    kept/docs/chars/reasons (the global near-dedup + holdout + stamp run in
    the parent over the completed shards)."""
    files, fname, out, prefix, rgmod, rgidx, cache_dir, dry = piece
    reject = reject_light if fname == "light" else reject_reason
    exact = set()
    os.makedirs(out, exist_ok=True)
    w = None if dry else ShardWriter(out, prefix)
    reasons = Counter()
    kept = docs = kept_chars = 0
    for spec in files:
        for text, url, src in iter_source(spec, cache_dir, rgmod, rgidx):
            text = SPECIAL_TOKEN.sub("", text).strip()
            docs += 1
            why = reject(text)
            if why is None:
                k = exact_key(text)
                if k in exact:
                    why = "exact_dup"
                else:
                    exact.add(k)
            reasons[why or "kept"] += 1
            if not why:
                kept += 1
                kept_chars += len(text)
                if w is not None:
                    w.write({"content": text, "source": src, "url": url})
            if docs % 100_000 == 0:
                # Workers inherit the parent's stdout (the launch log); without this
                # the parallel path is silent until pool.map returns and the launch
                # monitor's 10-min silence rule false-fails a healthy hours-long run.
                print(
                    f"[{prefix.rstrip('_')}] {docs} in | {reasons['kept']} kept | "
                    f"{kept_chars / CHARS_PER_TOKEN / 1e9:.2f}B tok",
                    flush=True,
                )
    if w is not None:
        w.close()
    return {"kept": kept, "docs": docs, "kept_chars": kept_chars, "reasons": dict(reasons)}


_CACHE = os.path.join(ROOT, "data", "raw")


def _worker_pieces(a):
    """: source specs -> list of per-worker slices (disjoint FILE ranges).
    Glob specs (jsonl:/parquet:) expand to their files; a named SOURCES spec
    goes to one worker whole (its files are downloaded one at a time and
    deleted, so two workers on one source would fight over the cache)."""
    import glob as _g

    specs = []
    for sp in a.source:
        if sp.startswith("jsonl:"):
            specs += [f"jsonl:{p}" for p in sorted(_g.glob(sp[len("jsonl:"):]))]
        elif sp.startswith("parquet:"):
            specs += [f"parquet:{p}" for p in sorted(_g.glob(sp[len("parquet:"):]))]
        else:
            specs.append(sp)
    N = a.workers
    slices = [[] for _ in range(N)]
    for i, sp in enumerate(specs):
        slices[i % N].append(sp)
    return [
        (s, a.filters, a.out, f"w{i}_", a.rg_mod, a.rg_idx, a.cache_dir, a.dry)
        for i, s in enumerate(slices)
        if s
    ]


def _global_pass(a):
    """: after parallel per-worker clean, remove global near-dups + holdout and
    stamp. Loads every shard, keeps only near-unique non-holdout docs, rewrites
    to a single prefix set (drop worker ids), writes build_corpus_stats + the
    filters_fp stamp over verified files.
    --global-only skips the worker pool and re-runs this pass over the existing
    w*_*.jsonl shards (a killed serial pass leaves them on disk)."""
    import glob as _g

    # Hold the build flock for the whole write; stamp time re-probes it and the
    # settle guard refuses if any other process writes under us (tilerl T7-2).
    global _LOCK_FD
    _LOCK_FD = _build_lock(a.out)

    paths = sorted(_g.glob(f"{a.out}/w*_*.jsonl"))
    reasons = Counter()
    if a.global_only:
        if not paths:
            raise SystemExit(f"REFUSE: --global-only but no w*_*.jsonl shards in {a.out}")
        # The merged output of a previous (killed) pass must not survive into this one.
        for p in _g.glob(f"{a.out}/{a.domain}_*.jsonl"):
            os.remove(p)
    else:
        # A previous run's leftovers would be read as this run's output. Wipe is the
        # operator's step (fb spec); the guard makes a missed wipe loud.
        stale = paths + _g.glob(f"{a.out}/w*_*.part")
        if stale:
            raise SystemExit(f"REFUSE: {len(stale)} worker shard(s) from a previous run in {a.out} -- wipe first")
        old = _g.glob(f"{a.out}/{a.domain}_*.jsonl")
        if old:
            raise SystemExit(f"REFUSE: {len(old)} '{a.domain}' shard(s) already in {a.out} -- wipe first")
        pieces = _worker_pieces(a)
        import multiprocessing as mp

        with mp.Pool(a.workers) as pool:
            res = pool.map(_clean_piece, pieces)
        reasons = Counter()
        for r in res:
            reasons.update(r["reasons"])
        paths = sorted(_g.glob(f"{a.out}/w*_*.jsonl"))
    total = sum(reasons.values()) or 1
    if a.dry:
        print(f"docs in {total} | kept {reasons['kept']} ({reasons['kept']/total:.1%}) [dry, workers={a.workers}]")
        for why, n in reasons.most_common():
            if why != "kept":
                print(f"  {why:18s} {n:9d}  {n/total:.1%}")
        return 0
    # global near-dedup + holdout over the kept shards (cross-worker dups)
    near = MinHashLSH(perms=128, bands=16)
    seen_exact = set()
    kept = kept_chars = 0
    # Worker-kept counts are pre-global; the pass below reclassifies some as
    # holdout/exact_dup/near_dup, so recount "kept" from its verdict.
    reasons["kept"] = 0
    w = ShardWriter(a.out, a.domain)
    for p in paths:
        print(f"global pass: {os.path.basename(p)}", flush=True)
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                t = d.get("content") or ""
                if not t:
                    continue
                # Holdout was already applied per doc in the worker phase (reject_light
                # ends in reject_holdout, per-line scan included); a doc reaching a w*
                # shard passed it on the same bytes. The global pass re-checks the whole
                # doc only, as the last gate before the stamp. The per-line scan is
                # O(lines) sha1s per doc and dominated this serial pass on code.
                if is_holdout(t) or is_holdout(QA_PREFIX.sub("", ANSWER_TAIL.split(t, 1)[0]).strip()):
                    reasons["holdout"] += 1
                    continue
                k = exact_key(t)
                if k in seen_exact:
                    reasons["exact_dup"] += 1
                    continue
                if not a.no_near_dedup and near.seen(t):
                    reasons["near_dup"] += 1
                    continue
                seen_exact.add(k)
                kept += 1
                kept_chars += len(t)
                reasons["kept"] += 1
                w.write(d)
    w.close()
    # drop the worker-id shards (superseded by the merged rewrite)
    for p in paths:
        os.remove(p)
    nshards = len(_g.glob(f"{a.out}/{a.domain}_*.jsonl"))
    total = sum(reasons.values()) or 1
    print(f"docs in {total} | kept {kept} ({kept/total:.1%}) | ~{kept_chars/CHARS_PER_TOKEN/1e9:.2f}B tok")
    for why, n in reasons.most_common():
        if why != "kept":
            print(f"  {why:18s} {n:9d}  {n/total:.1%}")
    _write_stats(a.out, a.domain, a, reasons, kept, kept_chars, nshards)
    return 0


def _build_lock(out):
    """Acquire an exclusive flock on {out}/.build.lock and record this PID. The
    returned fd must be HELD for the write (callers bind it to _LOCK_FD) or CPython
    close()s it on return and the flock is gone -- that was the defect tilerl-0a
    caught (T7-2 revote): production dropped the return and the lock never survived.
    flock gives mutual exclusion between two builds; the settle probe reads the PID
    rather than re-flocking, because flock is per-open-file-description and a second
    open() by the SAME process conflicts with its own lock (Defect B). Returns the fd."""
    import fcntl

    os.makedirs(out, exist_ok=True)
    f = open(os.path.join(out, ".build.lock"), "a+", encoding="utf-8")  # noqa: SIM115 -- fd is returned to hold the flock
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.close()
        raise SystemExit(f"REFUSE: another build holds {out}/.build.lock -- two writers on one domain") from None
    f.seek(0)
    f.truncate()
    f.write(str(os.getpid()))
    f.flush()
    return f


_LOCK_FD = None  # the held build-lock fd for the whole write; set by the pass entry points


def _other_writer_pid(out):
    """The PID of a live writer on {out}, or None if ours/free/stale. Reads the
    lock's PID and refuses to call it 'another writer' if it is THIS process or a
    dead PID (a crashed build leaves a stale PID; flock already released). This is
    the probe the settle guard and preflight share -- PID-based, not pgrep-name,
    and it does not re-flock (a re-flock by the same process would conflict with
    its own OFD)."""
    lock = os.path.join(out, ".build.lock")
    if not os.path.exists(lock):
        return None
    try:
        with open(lock, encoding="utf-8") as fh:
            pid = int(fh.read().strip())
    except (OSError, ValueError):
        return None
    if pid == os.getpid():
        return None  # our own build: it wrote this PID, it is about to stamp its own output
    try:
        os.kill(pid, 0)  # liveness probe: raises if no such process
        return pid
    except OSError:
        return None  # stale: the builder died, its flock is gone, the dir is free


def _settle_dir(out, domain, settle_s):
    """Stamp-time settle guard (tilerl T7-2): refuse the stamp unless the dir is
    not being written. Two mechanisms. First, PID-based (not pgrep-name):
    `_other_writer_pid` refuses if a LIVE, OTHER process holds the build lock.
    Second, the file set is read twice a beat apart: if any shard appears/disappears
    or its mtime CHANGES between the two reads, a writer is touching it and the
    stamp is refused. When the lock is ours (a real build holds it and is stamping
    the bytes it just finished), fresh mtimes are expected, so the settle-window
    mtime check applies only when NO other writer could be mid-write -- i.e.
    every shard must be older than `settle_s` ONLY when we do not hold our own lock
    (an ad-hoc re-stamp, or a non-locking foreign appender like the U+2028 repair
    that never took the lock)."""
    import time as _time

    who = _other_writer_pid(out)
    if who is not None:
        raise SystemExit(
            f"REFUSE: {out} is locked by live pid {who} (not ours) -- stamp over unsettled bytes"
        )
    lock = os.path.join(out, ".build.lock")
    self_hold = os.path.exists(lock)
    if self_hold:
        try:
            with open(lock, encoding="utf-8") as fh:
                self_hold = int(fh.read().strip()) == os.getpid()
        except (OSError, ValueError):
            self_hold = False

    def _mtime_below(path):
        try:
            return settle_s > 0 and (_time.time() - os.path.getmtime(path)) < settle_s
        except OSError:
            return True  # vanished mid-check: treat as unsettled

    def snap():
        return {p: os.path.getmtime(p) for p in sorted(glob.glob(os.path.join(out, f"{domain}_*.jsonl")))}

    s1 = snap()
    _time.sleep(0.5)
    s2 = snap()
    if not s1:
        raise SystemExit(f"REFUSE: {domain} has no {domain}_*.jsonl shards to stamp")
    changed = [k for k in set(s1) | set(s2) if s1.get(k) != s2.get(k)]
    if changed:
        raise SystemExit(f"REFUSE: {domain} shard set changed between two reads (mid-write): {os.path.basename(changed[0])}")
    if not self_hold and settle_s > 0 and any(_mtime_below(p) for p in s2):
        raise SystemExit(
            f"REFUSE: some {domain} shard was modified within the settle window "
            f"({settle_s}s) and no build holds the lock -- a writer may still be touching it"
        )


SETTLE_S = 60


def _write_stats(out, domain, a, reasons, kept, kept_chars, nshards):
    import sys as _sys

    _sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    from corpus_fingerprint import fp_dir as _fp_dir  # noqa: E402
    from corpus_fingerprint import fp_filters as _fp_filters  # noqa: E402

    # Settle guard (tilerl T7-2, 2026-09-01): refuse a stamp over a dir a writer
    # is still touching. Tonight's mid-write stamp (the U+2028 repair wrote tail
    # shards 19 min after the stamp/its fingerprint) is exactly this class. The
    # stamp's fingerprint must cover the SETTLED bytes, so compute it only after
    # the dir passes: no live writer (flock, PID-based not pgrep-name), every
    # shard's mtime older than the settle window, file set stable across two reads.
    _settle_dir(out, domain, SETTLE_S)

    stats = {
        "domain": domain, "reasons": dict(reasons), "kept": kept,
        "kept_chars": kept_chars, "kept_tokens": int(kept_chars / CHARS_PER_TOKEN),
        "filters": a.filters, "workers": a.workers, "n_shards": nshards,
        # One stamp over the verified shards: what produced them (filters) and
        # what they contain (dir). A domain without fingerprint fails the
        # mix-domain guard at training start.
        "filters_fp": _fp_filters(),
        "fingerprint": _fp_dir(out),
        # Whether the global near-dedup pass ran. A false stamp must say why on
        # the artifact: the stage-2 re-stamp prerequisite is recorded here, not
        # in a message (fb ruling 2026-08-31).
        "near_dedup": not a.no_near_dedup,
        "near_dedup_note": (
            None
            if not a.no_near_dedup
            else "skipped by the 2026-08-31 dedup ruling; replaced by the separate calibrated near-dedup post-pass (44), removed fraction lands as a fact"
        ),
    }
    # Measured tokens, counted the way training counts them. kept_tokens above is
    # a chars/1.5 estimate and stays for continuity; `tokens` is the number a mix
    # budget is read against. Both are stamped, so which is which is never a guess.
    tok_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "tokenizer.json")
    if os.path.exists(tok_path):
        try:
            from tokenizers import Tokenizer

            from count_tokens import CONVENTION, count_shards

            shards = sorted(glob.glob(os.path.join(out, f"{domain}_*.jsonl")))
            if shards:
                n_sample = min(3, len(shards))
                tokens, _ = count_shards(shards, Tokenizer.from_file(tok_path), sample=n_sample)
                stats["tokens"] = tokens
                stats["tokens_status"] = "measured"
                stats["tokens_config"] = (
                    f"data/tokenizer.json, {n_sample}/{len(shards)}-shard sample extrapolated by bytes; {CONVENTION}"
                )
        except Exception as e:  # a stamp must land even if the count cannot
            stats["tokens_status"] = f"unmeasured: {type(e).__name__}: {str(e)[:80]}"
    else:
        stats["tokens_status"] = "unmeasured: data/tokenizer.json not present"
    with open(os.path.join(out, "build_corpus_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)


def _run_workers(a, target_chars):
    # Parallel exact-dup + holdout global pass (fb ruling 2026-08-31): stage-2
    # consolidates already-cleaned w* shards; exact-dedup is hash-set and
    # parallelisable, the near-dedup MinHash is replaced by a later calibrated
    # pass (44), so stage-2 runs exact-only and stamps near_dedup:false.
    if a.global_only and a.no_near_dedup and a.workers > 1:
        return _parallel_exact_pass(a)
    return _global_pass(a)


def _emit_slice(args):
    """Phase A worker: emit (exact_key, base+ordinal) for non-holdout docs of one
    w* shard. Module-level so Pool can pickle it; uses only module globals."""
    _path, _base = args
    emit = []
    holdout = 0
    ords = 0
    with open(_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            t = SPECIAL_TOKEN.sub("", json.loads(line).get("content") or "").strip()
            if not t:
                continue
            if is_holdout(t) or is_holdout(QA_PREFIX.sub("", ANSWER_TAIL.split(t, 1)[0]).strip()):
                holdout += 1
                continue
            emit.append((exact_key(t), _base + ords))
            ords += 1
    return emit, holdout


def _parallel_exact_pass(a):
    """Two-phase global EXACT-dup + holdout, byte-identical to the serial pass.
    Phase A (parallel): globally-order the w* shards, for each non-holdout doc
    emit (exact_key, ordinal); Phase B (serial, compact): keep the min ordinal
    per key -> survivor ordinals (first occurrence, same rule the serial pass
    keeps); Phase C (parallel): re-read in the same global order and rewrite
    only survivor docs. The output equals serial exact+holdout byte-for-byte."""
    import glob as _g
    import multiprocessing as mp

    global _LOCK_FD
    _LOCK_FD = _build_lock(a.out)  # hold the build flock through the write; stamp refuses interference
    paths = sorted(_g.glob(f"{a.out}/w*_*.jsonl"))
    if not paths:
        raise SystemExit(f"REFUSE: no w*_*.jsonl shards in {a.out}")
    for p in _g.glob(f"{a.out}/{a.domain}_*.jsonl"):
        os.remove(p)

    # Global ordinal per doc: slice bases are prefix sums of per-shard doc counts,
    # so phase A is embarrassingly parallel and phase C restores the same order.
    doc_counts = []
    for p in paths:
        n = 0
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    n += 1
        doc_counts.append(n)
    bases = []
    acc = 0
    for n in doc_counts:
        bases.append(acc)
        acc += n
    total_docs = acc

    with mp.Pool(a.workers) as pool:
        phase_a = pool.map(_emit_slice, [(p, bases[i]) for i, p in enumerate(paths)])

    # Phase B: survivor = min ordinal per exact key (first occurrence)
    seen = {}
    for emit, _ in phase_a:
        for k, o in emit:
            if k not in seen or o < seen[k]:
                seen[k] = o
    survivor = set(seen.values())
    exact_dups = total_docs - len(seen)
    holdout_total = sum(h for _, h in phase_a)

    # Phase C: re-read and write survivors, in global order.
    w = ShardWriter(a.out, a.domain)
    kept = kept_chars = 0
    reasons = Counter()
    for i, p in enumerate(paths):
        ords = 0
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                t = SPECIAL_TOKEN.sub("", d.get("content") or "").strip()
                if not t:
                    continue
                if is_holdout(t) or is_holdout(QA_PREFIX.sub("", ANSWER_TAIL.split(t, 1)[0]).strip()):
                    continue  # holdout counted in A; skip in rewrite
                if (bases[i] + ords) in survivor:
                    kept += 1
                    kept_chars += len(t)
                    reasons["kept"] += 1
                    w.write(d)
                ords += 1
    w.close()
    for p in paths:
        os.remove(p)
    nshards = len(_g.glob(f"{a.out}/{a.domain}_*.jsonl"))
    reasons["holdout"] = holdout_total
    reasons["exact_dup"] = exact_dups
    _write_stats(a.out, a.domain, a, reasons, kept, kept_chars, nshards)
    return 0


def _word_shingle_hashes(normalised_text, shingle=3):
    """Word-n-gram hashes of a normalised doc, distinct and order-stable. blake2b is
    fixed-reproducible (independent of PYTHONHASHSEED), so serial and parallel workers
    emit identical shingle sets and the rewrite is byte-identical. The near-dup
    DECISION is exact Jaccard over THIS set (normalized word-3-gram, near_dedup_gate.md);
    MinHash only generates candidates. () for docs under `shingle` words: nothing to dedup."""
    words = normalised_text.split()
    if len(words) < shingle:
        return ()
    seen, out = set(), []
    for i in range(len(words) - shingle + 1):
        h = int.from_bytes(
            hashlib.blake2b(" ".join(words[i : i + shingle]).encode("utf-8"), digest_size=8).digest(), "little"
        )
        if h not in seen:
            seen.add(h)
            out.append(h)
    return tuple(out)


def _near_coeffs(perms, seed):
    """(ab, mask): `perms` deterministic permutation coefficients (seed), the same
    construction MinHashLSH uses. Fixed seed => identical signatures serial vs parallel."""
    import random as _r  # noqa

    rng = _r.Random(seed)
    mask = (1 << 61) - 1
    return [(rng.randrange(1, mask), rng.randrange(0, mask)) for _ in range(perms)], mask


def _minhash(shingle_hashes, ab, mask):
    """MinHash signature (MIN over each permutation of the shingle-hash set). Candidate
    generator only; a signature asserts nothing about near-ness by itself."""
    return tuple(min(((a * h + b) & mask) for h in shingle_hashes) for a, b in ab)


def _norm_skeleton(t):
    """Default post-pass normaliser: lowercase + collapse whitespace. The real
    per-domain code/en_c4/math_owm normalisers (with 44's calibrated keyword stoplist
    and LaTeX mapping) are injected per domain at run time; this keeps the core
    engine testable and normaliser-agnostic while the domain constants are
    calibration artifacts (bake the gate's margins)."""
    return " ".join(t.lower().split())


def _lsh_candidates(sigs, bands, rows):
    """Candidate ordinal pairs from MinHash LSH: two docs sharing any band's full
    row-tuple are candidates. False candidates cost compute only -- the exact-J
    decision below filters them and never errs upward."""
    rivals = set()
    for b in range(bands):
        table = {}
        for o, sig in sigs.items():
            table.setdefault(sig[b * rows : (b + 1) * rows], []).append(o)
        for members in table.values():
            if len(members) > 1:
                members.sort()
                for i in range(len(members)):
                    for j in range(i + 1, len(members)):
                        rivals.add((members[i], members[j]))
    return rivals


def _lsh_recall_bound(bands, rows, jac):
    """MinHash LSH collision probability at Jaccard `jac`: 1-(1-jac**rows)**bands. The
    THEORETICAL recall -- the acceptance basis at 8M-doc scale, where the 50-doc
    hand-read (1225 pairs) cannot empirically cover recall (44 condition 1). epsilon =
    1 - bound at the gate threshold is a second error term of the removed-fraction fact,
    beside the Wilson CI (44 condition 1; a missed pair here survives as a near-dup)."""
    return 1.0 - (1.0 - jac**rows) ** bands


def _union_find(n):
    """Disjoint-set over 0..n-1 with path-halving + rank union. Deterministic: cluster
    assignment depends only on the unordered set of union() calls, so serial and
    parallel runs compute identical survivors (kept = min ordinal per cluster)."""
    parent = list(range(n))
    rank = [0] * n

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1

    return find, union


def _near_emit_slice(args):
    """Phase A worker: (path, base, ab, mask, normaliser) -> list[(ordinal, sig)].
    Ordinal = position among non-empty content lines in global shard order (identical
    walk in the rewrite); sig = MinHash or None for docs too short to shingle (always
    survive). The stamped input is already holdout- and exact-deduped, so near-dup is
    the only axis. Module-level for mp.Pool pickling."""
    path, base, ab, mask, norm = args
    emit, ords = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            t = SPECIAL_TOKEN.sub("", json.loads(line).get("content") or "").strip()
            if not t:
                continue
            sh = _word_shingle_hashes(norm(t))
            emit.append((base + ords, _minhash(sh, ab, mask) if sh else None))
            ords += 1
    return emit


def _near_write_stats(out, domain, reasons, kept, kept_chars, nshards, removed_n, total_docs, recall, cfg):
    """Stamp the post-pass output like _write_stats (fingerprint triad + tokens), but
    carry the near-dedup configuration and result: they are part of the artifact's
    meaning, and the removed-fraction fact reads them off the stamp."""
    import sys as _sys

    _sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    from corpus_fingerprint import fp_dir as _fp_dir  # noqa: E402
    from corpus_fingerprint import fp_filters as _fp_filters  # noqa: E402

    _settle_dir(out, domain, SETTLE_S)
    stats = {
        "domain": domain, "reasons": dict(reasons), "kept": kept,
        "kept_chars": kept_chars, "kept_tokens": int(kept_chars / CHARS_PER_TOKEN),
        "filters": "near-dedup-postpass", "n_shards": nshards,
        "filters_fp": _fp_filters(),
        "fingerprint": _fp_dir(out),
        "near_dedup": True,
        "removed_fraction": (removed_n / total_docs) if total_docs else 0.0,
        "recall_bound": recall,
        "recall_epsilon": 1.0 - recall,
        "config": cfg,
    }
    tok_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "tokenizer.json")
    if os.path.exists(tok_path):
        try:
            from tokenizers import Tokenizer  # noqa: I001 -- same nested-import shape as _write_stats:622

            from count_tokens import CONVENTION, count_shards

            shards = sorted(glob.glob(os.path.join(out, f"{domain}_*.jsonl")))
            if shards:
                n_sample = min(3, len(shards))
                tokens, _ = count_shards(shards, Tokenizer.from_file(tok_path), sample=n_sample)
                stats["tokens"] = tokens
                stats["tokens_status"] = "measured"
                stats["tokens_config"] = f"{n_sample}/{len(shards)}-shard sample extrapolated by bytes; {CONVENTION}"
        except Exception as e:
            stats["tokens_status"] = f"unmeasured: {type(e).__name__}: {str(e)[:80]}"
    else:
        stats["tokens_status"] = "unmeasured: data/tokenizer.json not present"
    with open(os.path.join(out, "build_corpus_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)


def _near_dedup_postpass(a, normaliser=None, perms=128, bands=8, rows=16, jaccard=0.5, seed=17):
    """Calibrated near-dedup post-pass (fb ruling 2026-08-31; 44 conditions 2026-09-01).
    Runs AFTER the stage-2 run finished (never launch-parallel); rewrites the stamped
    {domain}_*.jsonl in place (new corpus_fp, forces retokenize). Byte-identical to a
    single-worker run of the same pipeline.

    Fork C (memory-bounded): Phase A keeps only each doc's MinHash signature in memory
    (1GB at 8M docs -- NOT the ~40GB of shingle sets), LSH yields candidate pairs, and
    the exact normalized word-3-gram Jaccard DECISION is made by re-reading only the
    candidate docs' source lines. Exact-J >= jaccard keeps the removed-fraction fact on
    44's calibrated measure: MinHash/LSH is a candidate generator, never a decider."""
    import multiprocessing as mp

    if bands * rows != perms:
        raise ValueError(f"bands*rows ({bands}x{rows}) != perms ({perms})")
    global _LOCK_FD
    _LOCK_FD = _build_lock(a.out)
    normaliser = normaliser or _norm_skeleton
    ab, mask = _near_coeffs(perms, seed)

    paths = sorted(glob.glob(f"{a.out}/{a.domain}_*.jsonl"))
    if not paths:
        raise SystemExit(f"REFUSE: no {a.domain}_*.jsonl shards to near-dedup in {a.out}")

    doc_counts = []
    for p in paths:
        n = 0
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    n += 1
        doc_counts.append(n)
    bases, acc = [], 0
    for n in doc_counts:
        bases.append(acc)
        acc += n
    total_docs = acc

    with mp.Pool(a.workers) as pool:
        phase_a = pool.map(_near_emit_slice, [(p, bases[i], ab, mask, normaliser) for i, p in enumerate(paths)])
    sigs = {}
    for emit in phase_a:
        for o, sig in emit:
            if sig is not None:
                sigs[o] = sig

    removed = set()
    rivals = _lsh_candidates(sigs, bands, rows)
    if rivals:
        wanted = set()
        for lo, hi in rivals:
            wanted.add(lo)
            wanted.add(hi)
        import bisect

        by_shard = {}
        for o in wanted:
            by_shard.setdefault(bisect.bisect_right(bases, o) - 1, set()).add(o)
        shingle = {}
        for i, p in enumerate(paths):
            cur = by_shard.get(i)
            if not cur:
                continue
            off = 0
            with open(p, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    t = SPECIAL_TOKEN.sub("", json.loads(line).get("content") or "").strip()
                    if bases[i] + off in cur and t:
                        shingle[bases[i] + off] = _word_shingle_hashes(normaliser(t))
                    if t:
                        off += 1
        find, union = _union_find(total_docs)
        for lo, hi in rivals:
            A = shingle.get(lo)
            B = shingle.get(hi)
            if not A or not B:
                continue
            sa, sb = set(A), set(B)
            j = len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0
            if j >= jaccard:
                union(lo, hi)
        roots = {}
        for o in sigs:
            roots.setdefault(find(o), []).append(o)
        for members in roots.values():
            if len(members) > 1:
                removed |= set(members) - {min(members)}

    kept_flags = bytearray(b"\x01") * total_docs  # 8M flags ~8MB, not a set
    for o in removed:
        kept_flags[o] = 0

    # rewrite survivors in global order into a staging dir, then move over originals.
    nd = os.path.join(a.out, f".near_{a.domain}")
    os.makedirs(nd, exist_ok=True)
    w = ShardWriter(nd, a.domain)
    kept = kept_chars = 0
    for i, p in enumerate(paths):
        ords = 0
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                t = SPECIAL_TOKEN.sub("", json.loads(line).get("content") or "").strip()
                if not t:
                    continue
                if kept_flags[bases[i] + ords]:
                    kept += 1
                    kept_chars += len(t)
                    w.write(json.loads(line))
                ords += 1
    w.close()
    for p in paths:
        os.remove(p)
    for sp in sorted(glob.glob(f"{nd}/{a.domain}_*.jsonl")):
        os.replace(sp, os.path.join(a.out, os.path.basename(sp)))
    os.rmdir(nd)

    removed_n = total_docs - kept
    recall = _lsh_recall_bound(bands, rows, jaccard)
    nshards = len(glob.glob(f"{a.out}/{a.domain}_*.jsonl"))
    reasons = Counter({"kept": kept, "near_dup": removed_n})
    _near_write_stats(
        a.out, a.domain, reasons, kept, kept_chars, nshards, removed_n, total_docs, recall,
        {"perms": perms, "bands": bands, "rows": rows, "jaccard": jaccard, "seed": seed},
    )
    return 0


def _ladder_frozen_domains():
    """Domain names any mix_scale_*.json freezes. Writing into one is a silent
    fingerprint break (2026-08-31: ten shards into data/corpus/code/); new corpus
    goes to a fresh dir. Read from the live mixes so a rename self-heals."""
    frozen = set()
    for p in glob.glob(os.path.join(ROOT, "data", "mix_scale_*.json")):
        try:
            with open(p) as f:
                frozen |= set(json.load(f).get("domains", {}).keys())
        except (OSError, ValueError):
            continue
    return frozen


def _preflight(a):
    """Refuse a clean that will waste the run (fb order 2026-08-31). Four gates,
    each independent and each with a failing-case --selftest:
      (a) no other live writer holds this domain        unique writer by pgrep
      (b) output dir exists and is not a ladder-frozen name
      (c) filter family matches the domain family       the web chain deletes code
      (d) a 1000-doc sample passes end-to-end           wrong filters for this source
    SystemExit on the first refusal. Run once, at start, before any fetch or clean."""
    out = os.path.normpath(os.path.join(OUT_DIR, a.domain) if a.out is None else a.out)
    base = os.path.basename(out)
    # (b) not a ladder-frozen name, and the dir exists
    if base in _ladder_frozen_domains():
        raise SystemExit(
            f"REFUSE: {out} is a ladder-frozen domain {base!r}; write new corpus to a fresh dir"
        )
    if not a.dry and not os.path.isdir(out):
        raise SystemExit(f"REFUSE: output dir {out} does not exist; mkdir is the operator's step")
    # (a) unique writer, PID-based (not pgrep-name): a live build records its PID in
    # {out}/.build.lock. `_other_writer_pid` refuses only if a LIVE, OTHER process
    # wrote it; our own (or a stale/crashed) lock is free. flock stays the write-time
    # mutual exclusion; this probe deliberately does not re-flock (a same-process
    # re-flock would conflict with the build's own held lock, Defect B).
    who = _other_writer_pid(out)
    if who is not None:
        raise SystemExit(
            f"REFUSE: another build is writing {a.domain} (live pid {who} holds {out}/.build.lock)"
        )
    # (c) filter family matches the domain family
    CODE_DOMAINS = {"code", "code_rp1t", "en", "en_c4", "math", "math_owm", "cot"}
    if a.domain in CODE_DOMAINS and a.filters != "light":
        raise SystemExit(
            f"REFUSE: domain {a.domain} is code-family; --filters must be light, got {a.filters}"
        )
    # (d) a 1000-doc sample passes end-to-end: 0-keep or a single reject reason
    # dominating the sample means the filter set is wrong for this source. The
    # existing REJECT_EARLY_AT catches it mid-run at 20k docs; this is the same
    # check moved to the start on a 1000-doc sample (cheap, no fetch for jsonl).
    _sample_ok(a, out)
    return out


def _sample_ok(a, out):
    """Run the filter chain + exact dedup on up to 1000 input docs; refuse if none
    keep or one reject reason dominates. --global-only samples the already-cleaned
    w* shards, so it normally passes; the check targets a fresh clean's raw input."""
    reject = reject_light if a.filters == "light" else reject_reason
    seen = 0
    kept = 0
    reasons = Counter()
    exact = set()
    if a.global_only:
        paths = sorted(glob.glob(os.path.join(out, "w*_*.jsonl")))
    else:
        paths = []
        for spec in a.source:
            g = spec[6:] if spec.startswith("jsonl:") else spec[8:] if spec.startswith("parquet:") else spec
            paths += sorted(glob.glob(g)) if ("jsonl:" in spec or "parquet:" in spec) else []
    for p in paths:
        for text, _ in iter_jsonl(p):
            text = SPECIAL_TOKEN.sub("", text).strip()
            if not text:
                continue
            seen += 1
            why = reject(text)
            if why is None:
                k = exact_key(text)
                if k in exact:
                    why = "exact_dup"
                else:
                    exact.add(k)
            reasons[why or "kept"] += 1
            if not why:
                kept += 1
            if seen >= 1000:
                break
        if seen >= 1000:
            break
    if not seen:
        return  # no input sampled; nothing to judge (a later gate handles empty input)
    if kept == 0:
        dom = reasons.most_common(1)
        raise SystemExit(
            f"REFUSE: {seen}-doc sample kept 0 (top reject {(dom[0][0], dom[0][1]) if dom else '?'}); "
            f"filters {a.filters!r} are wrong for this source -- check --filters/--source"
        )
    if not a.global_only:
        top, topn = reasons.most_common(1)[0]
        if top != "kept" and topn / seen > 0.95:
            raise SystemExit(
                f"REFUSE: {seen}-doc sample rejects {topn}/{seen} as {top!r}; "
                f"filters {a.filters!r} are wrong for this source"
            )


def _selftest_preflight():
    """Failing-case worlds for the four gates: each must REFUSE. Broken worlds
    mutate a real artifact (the ladder mix and a live pgrep line), per harness."""
    import tempfile

    ok = 0

    class A:  # a fake argparse namespace
        def __init__(self, **kw):
            self.__dict__.update(kw)

    # (b) frozen-name output -> REFUSE
    frozen = _ladder_frozen_domains()
    assert frozen, "selftest needs at least one ladder-frozen domain"
    name = sorted(frozen)[0]
    try:
        _preflight(A(domain=name, out=None, filters="light", dry=True, source=["jsonl:/dev/null"], global_only=False))
        raise AssertionError("(b) frozen output name did not REFUSE")
    except SystemExit:
        ok += 1
    # (c) code-family domain with wrong filters -> REFUSE
    try:
        _preflight(A(domain="en_c4", out=None, filters="web", dry=True, source=["jsonl:/dev/null"], global_only=False))
        raise AssertionError("(c) code-family with web filters did not REFUSE")
    except SystemExit:
        ok += 1
    # (a) + T7-2: a FOREIGN live pid in the build lock -> REFUSE; our OWN lock -> PASS.
    # tilerl-0a's rejection: the old case held the lock through THIS process, which
    # the new PID probe must (correctly) NOT refuse -- the bug was testing a shape
    # production does not take. A foreign live pid is the real duplicate-writer.
    import subprocess as _sp

    src = tempfile.mkdtemp()
    marker = os.path.join(src, "held_domain")
    os.makedirs(marker)
    with open(os.path.join(marker, "x_000.jsonl"), "w") as fh:
        fh.write(json.dumps({"content": "x" * 500, "url": "u"}) + "\n")
    # A live process (+ a second we kill) for the writer states. tilerl-0a's
    # call-shape-fidelity catch (2026-09-01): the previous selftest passed
    # `settle_s=0`, which short-circuits the mtime-window branch (`settle_s > 0`)
    # that this guard exists for. Production stamps with SETTLE_S=60, so every
    # settle case here uses SETTLE_S; cases 3 and 4 (stale lock + fresh shard,
    # no lock + fresh shard) are the two real shapes nothing covered before.
    def _refuses(marker, tag):
        try:
            _settle_dir(marker, "x", SETTLE_S)
            raise AssertionError(f"{tag}: did not REFUSE")
        except SystemExit:
            return 1

    sleeper = _sp.Popen(["sleep", "300"])  # a live process with a foreign pid
    dead = _sp.Popen(["sleep", "300"])  # killed below: a stale pid, not foreign-live
    dead_pid = str(dead.pid)
    dead.kill()
    dead.wait()
    try:
        # (a) foreign live pid in the lock -> preflight refuses (duplicate writer)
        with open(os.path.join(marker, ".build.lock"), "w", encoding="utf-8") as fh:
            fh.write(str(sleeper.pid))
        try:
            _preflight(A(domain="held_domain", out=marker, filters="light", dry=True,
                         source=[f"jsonl:{os.path.join(marker, 'x.jsonl')}"], global_only=False))
            raise AssertionError("(a) duplicate writer did not REFUSE")
        except SystemExit:
            ok += 1
        # (1) our OWN held build lock -> settle must PASS: production stamps its own
        #     fresh bytes, and the mtime window is skipped when the lock is ours.
        os.remove(os.path.join(marker, ".build.lock"))
        own = _build_lock(marker)  # we are the writer
        _settle_dir(marker, "x", SETTLE_S)  # self lock must NOT refuse (PASS, case 1)
        own.close()
        # (2) foreign live pid in the lock -> settle refuses via the PID probe
        with open(os.path.join(marker, ".build.lock"), "w", encoding="utf-8") as fh:
            fh.write(str(sleeper.pid))
        ok += _refuses(marker, "(2) foreign live writer")
        # (3) stale lock (dead pid) + fresh shard -> settle REFUSES: the PID probe
        #     sees no live writer, but the shard is fresh and no live build holds
        #     the lock, so the mtime window must catch it (a crashed build).
        with open(os.path.join(marker, ".build.lock"), "w", encoding="utf-8") as fh:
            fh.write(dead_pid)
        ok += _refuses(marker, "(3) stale lock over fresh shard")
        # (4) NO lock + fresh shard -> settle REFUSES: the U+2028 repair's exact
        #     shape -- a non-locking appender wrote shards 19 min after the stamp.
        #     Settle must catch it via the mtime window; this is the guard's reason
        #     to exist and needs settle_s > 0 to fire.
        os.remove(os.path.join(marker, ".build.lock"))
        ok += _refuses(marker, "(4) lockless writer over fresh shard")
    finally:
        sleeper.kill()
        sleeper.wait()
        import shutil
        shutil.rmtree(src, ignore_errors=True)
    # (d) a source whose 1000-doc sample keeps 0 (all <100 chars under light) -> REFUSE
    bad = tempfile.mkdtemp()
    bf = os.path.join(bad, "junk.jsonl")
    with open(bf, "w") as fh:
        for _ in range(50):
            fh.write(json.dumps({"content": "hi" * 5, "url": "u"}) + "\n")  # 10 chars, <100 -> short
    try:
        _preflight(A(domain="bad_dom", out=bad, filters="light", dry=True,
                     source=[f"jsonl:{bf}"], global_only=False))
        raise AssertionError("(d) 0-kept sample did not REFUSE")
    except SystemExit:
        ok += 1
    shutil.rmtree(bad, ignore_errors=True)
    # (the settle cases 1-4 + foreign-live-pid refuse live with gate (a) above)
    print(f"build_corpus selftest OK: {ok} gates refuse on their failing world (incl. T7-2 settle 2/3/4 + self-hold PASS)")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", action="append", default=None, help="fineweb2 | skypile | jsonl:<glob>")
    ap.add_argument("--target_tokens", type=float, default=8e9)
    ap.add_argument("--domain", default=None, help="a default-mix domain -> data/corpus/<domain>/")
    ap.add_argument("--out", default=None, help="output dir (default data/corpus/<domain>)")
    ap.add_argument(
        "--host_cap",
        type=int,
        default=20_000,
        help="max docs per URL host. This is a WEB-CRAWL filter: it stops one site from "
        "dominating a scrape. A single-source corpus is all one host, so the default "
        "silently discards most of it -- Chinese Wikipedia lost 83.4%% to it (192,417 of "
        "230,792 documents) before this note existed. Pass 0 to disable for such a source.",
    )
    ap.add_argument("--limit", type=int, default=None, help="stop after N input docs (dry runs)")
    ap.add_argument("--dry", action="store_true", help="no output files, print the rejects histogram")
    ap.add_argument(
        "--filters",
        choices=("web", "light"),
        default="web",
        help="web = the full Chinese-web filter chain; light = length/bytes/holdout only",
    )
    ap.add_argument(
        "--no_near_dedup",
        action="store_true",
        help="skip MinHash near-dedup (~30ms/doc, pure python) for already-deduped sources",
    )
    ap.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="jsonl glob whose documents are pre-seeded into the dedup set, so a domain "
        "built earlier is not repeated inside this one",
    )
    ap.add_argument("--cache_dir", default=os.path.join(ROOT, "data", "raw"))
    ap.add_argument(
        "--rg_mod",
        type=int,
        default=None,
        help="parallelize one parquet source: process only row groups i with i %% rg_mod == rg_idx",
    )
    ap.add_argument(
        "--rg_idx",
        type=int,
        default=None,
        help="parallel row-group index; run the same build with --rg_mod K for i in 0..K-1",
    )
    ap.add_argument("--workers", type=int, default=1, help="parallel clean: N workers over disjoint source-file slices; worker id in the shard prefix; global near-dup+holdout+stamp at the end")
    ap.add_argument("--global-only", action="store_true", help="skip the worker phase; re-run the global pass over existing w*_*.jsonl shards (a killed serial pass leaves them)")
    ap.add_argument("--selftest", action="store_true", help="run the pre-flight failing-case worlds and exit")
    a = ap.parse_args()
    if a.selftest:
        return _selftest_preflight()
    if not a.source or not a.domain:
        ap.error("--source and --domain are required (or pass --selftest)")
    a.out = a.out or os.path.join(OUT_DIR, a.domain)
    # Pre-flight: refuse a clean that will waste the run before any fetch or
    # worker starts (unique writer, non-frozen existing output dir, filter-family
    # match, 1000-doc sample). Raises SystemExit on the first failure.
    _preflight(a)

    target_chars = a.target_tokens * CHARS_PER_TOKEN
    reasons, hosts = Counter(), Counter()
    exact, near = set(), MinHashLSH()
    reject = reject_light if a.filters == "light" else reject_reason
    for pat in a.exclude:
        n0 = len(exact)
        for path in sorted(glob.glob(pat)):
            for text, _ in iter_jsonl(path):
                exact.add(exact_key(text))
        print(f"exclude {pat}: dedup set +{len(exact) - n0} docs", flush=True)
    if a.workers > 1 or a.global_only:
        # Loud refusals: a silently ignored flag here is a kept-set divergence
        # from the single-worker run that nothing reports.
        if a.limit:
            raise SystemExit("--limit is global-docs-seen; not supported with --workers (per-worker would change the kept set)")
        if a.exclude:
            raise SystemExit("--exclude is single-worker only; per-worker exclude seeding is not implemented")
        if a.rg_mod:
            raise SystemExit("--rg_mod and --workers are two parallelisms; pick one")
        return _run_workers(a, target_chars)
    kept_chars = seen = 0
    writers = {}
    for spec in a.source:
        for text, url, src in iter_source(spec, a.cache_dir, a.rg_mod, a.rg_idx):
            text = SPECIAL_TOKEN.sub("", text).strip()
            seen += 1
            if a.limit and seen > a.limit:
                break
            why = reject(text)
            if why is None:
                host = urlsplit(url).netloc if url else ""
                if host and a.host_cap and hosts[host] >= a.host_cap:
                    why = "host_cap"
                else:
                    k = exact_key(text)
                    if k in exact:
                        why = "exact_dup"
                    elif not a.no_near_dedup and near.seen(text):
                        why = "near_dup"
                    else:
                        exact.add(k)
            reasons[why or "kept"] += 1
            # (b) fast-fail: if the first REJECT_EARLY_AT docs reject >95% under
            # a single reason, the filter chain is wrong for this source (the
            # not_zh-0-kept wall surfaced after 3.96M docs). Stop in seconds,
            # name the reason, don't burn an hour.
            if seen == REJECT_EARLY_AT:
                n = sum(reasons.values()) or 1
                if reasons["kept"] / n < 0.05 and reasons:
                    top, topn = reasons.most_common(1)[0]
                    raise SystemExit(
                        f"FAST-FAIL after {seen} docs: {(reasons['kept']) / n:.0%} kept, "
                        f"~{topn / n:.0%} rejected as '{top}'. The filter set is wrong "
                        f"for this source -- check --filters."
                    )
            if why:
                continue
            if host:
                hosts[host] += 1
            kept_chars += len(text)
            if not a.dry:
                writers.setdefault(src, ShardWriter(a.out, src)).write(
                    {"content": text, "source": src, "url": url}
                )
            if seen % 100_000 == 0:
                print(
                    f"{seen} in | {reasons['kept']} kept | {kept_chars / CHARS_PER_TOKEN / 1e9:.2f}B tok",
                    flush=True,
                )
            if kept_chars >= target_chars:
                break
        if kept_chars >= target_chars:
            break
    for w in writers.values():
        w.close()
    total = sum(reasons.values()) or 1
    print(
        f"docs in {total} | kept {reasons['kept']} ({reasons['kept'] / total:.1%}) | "
        f"~{kept_chars / CHARS_PER_TOKEN / 1e9:.2f}B tokens"
    )
    for why, n in reasons.most_common():
        if why != "kept":
            print(f"  {why:18s} {n:9d}  {n / total:.1%}")
    if not a.dry:
        # A 0-kept domain must raise: an empty source glob surfaces later as a
        # missing cache with no hint why.
        if reasons["kept"] == 0:
            raise SystemExit(
                f"ERROR: domain '{a.domain}' kept 0 documents from {a.source} -- "
                f"check the --source glob / HF prefix (nothing written to {a.out})"
            )
        # Corpus fingerprint: the canonical implementation, not a copy -- a stamper that
        # diverged from the guard would stamp ids the guard never recognizes.
        import sys as _sys

        _sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
        from corpus_fingerprint import fp_dir as _fp_dir  # noqa: E402
        from corpus_fingerprint import fp_filters as _fp_filters  # noqa: E402

        with open(os.path.join(a.out, "build_corpus_stats.json"), "w") as f:
            json.dump(
                {
                    "reasons": reasons,
                    "top_hosts": hosts.most_common(50),
                    "fingerprint": _fp_dir(a.out),
                    # What produced this corpus, not just what it contains: the same Build
                    # command before and after a filters/ edit yields different shards, and
                    # PROVENANCE records only the command.
                    "filters_fp": _fp_filters(),
                },
                f,
                ensure_ascii=False,
                indent=1,
            )


if __name__ == "__main__":
    main()
