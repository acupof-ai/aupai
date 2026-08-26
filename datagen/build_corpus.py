#!/usr/bin/env python3
"""Build the pretraining corpus: select, clean, dedup, cap — the source is interchangeable.

    python datagen/build_corpus.py --source fineweb2 --target_tokens 8e9
    python datagen/build_corpus.py --source jsonl:data/raw/*.jsonl --dry --limit 2000   # inspect rejects

Sources (any mix, repeatable): fineweb2 (HuggingFaceFW/fineweb-2, zho_Hans), skypile
(Skywork/SkyPile-150B), jsonl:<glob> (rows with "content"/"text" [+ "url"]). Shards are streamed one at a
time, so disk holds one parquet at once. Output: data/corpus/primary/<source>_NNN.jsonl (100MB shards,
{"content","source","url"}) which train.load_texts() already reads, plus a rejects histogram.

Filters (order = cost): length, CJK ratio, bad bytes, symbol/digit ratio, boilerplate markers, URL
density, line structure (nav menus, duplicated lines), unfinished tail, garbage topics (filters/pass*),
eval-holdout contamination, exact dedup, MinHash near-dedup, per-host cap.
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
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from holdout import is_holdout  # noqa: E402

OUT_DIR = os.path.join(ROOT, "data", "corpus", "primary")
SHARD_BYTES = 100 * 2**20
CHARS_PER_TOKEN = 1.5

SOURCES = {
    "fineweb2": ("HuggingFaceFW/fineweb-2", "data/zho_Hans/train/", "text", "url"),
    "skypile": ("Skywork/SkyPile-150B", "data/", "text", None),
}

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
    pats = []
    for name in ("pass1_garbage", "pass2_garbage", "pass3_garbage"):
        path = os.path.join(ROOT, "filters", f"{name}.py")
        if os.path.exists(path):
            ns = {}
            exec(compile(open(path, encoding="utf-8").read(), path, "exec"), ns)  # module-level PATTERNS only
            pats += ns.get("PATTERNS", [])
    return re.compile("|".join(f"(?:{p})" for p in pats)) if pats else None


GARBAGE = load_garbage_patterns()


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
    if n < 2000 and not END_OK.search(text):
        return "unfinished"
    if GARBAGE is not None and GARBAGE.search(text[:600]):
        return "garbage_topic"
    for ln in lines:
        if 15 <= len(ln) <= 200 and is_holdout(ln):
            return "eval_contaminated"
    return None


# ---- dedup -------------------------------------------------------------------------------------
_NORM = re.compile(r"[\s\W_]+", re.UNICODE)


def exact_key(text):
    return hashlib.sha1(_NORM.sub("", text).encode("utf-8")).digest()[:12]


class MinHashLSH:
    """128-perm MinHash over char 5-gram shingles, 16 bands x 8 rows (~0.8 Jaccard threshold).
    ponytail: pure-python, in-memory; fine to ~30M docs, switch to datasketch/rensa beyond that."""

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
    with op(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                yield d.get("content") or d.get("text") or "", d.get("url")


def iter_source(spec, cache_dir):
    """Yield (text, url, source_name)."""
    if spec.startswith("jsonl:"):
        for p in sorted(glob.glob(spec[6:])):
            for text, url in iter_jsonl(p):
                yield text, url, os.path.basename(p).split(".")[0]
        return
    repo, prefix, text_col, url_col = SOURCES[spec]
    from huggingface_hub import hf_hub_download, list_repo_files

    files = sorted(f for f in list_repo_files(repo, repo_type="dataset") if f.startswith(prefix))
    for f in files:
        local = hf_hub_download(repo, f, repo_type="dataset", local_dir=cache_dir)  # real file, removable
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

    def write(self, row):
        if self.f is None or self.size >= SHARD_BYTES:
            if self.f:
                self.f.close()
            self.f = open(
                os.path.join(self.out_dir, f"{self.prefix}_{self.n:03d}.jsonl"), "w", encoding="utf-8"
            )
            self.n, self.size = self.n + 1, 0
        line = json.dumps(row, ensure_ascii=False) + "\n"
        self.f.write(line)
        self.size += len(line.encode("utf-8"))

    def close(self):
        if self.f:
            self.f.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", action="append", required=True, help="fineweb2 | skypile | jsonl:<glob>")
    ap.add_argument("--target_tokens", type=float, default=8e9)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--host_cap", type=int, default=20_000, help="max docs per URL host")
    ap.add_argument("--limit", type=int, default=None, help="stop after N input docs (dry runs)")
    ap.add_argument("--dry", action="store_true", help="no output files, print the rejects histogram")
    ap.add_argument("--cache_dir", default=os.path.join(ROOT, "data", "raw"))
    a = ap.parse_args()

    target_chars = a.target_tokens * CHARS_PER_TOKEN
    reasons, hosts = Counter(), Counter()
    exact, near = set(), MinHashLSH()
    kept_chars = seen = 0
    writers = {}
    for spec in a.source:
        for text, url, src in iter_source(spec, a.cache_dir):
            seen += 1
            if a.limit and seen > a.limit:
                break
            why = reject_reason(text)
            if why is None:
                host = urlsplit(url).netloc if url else ""
                if host and hosts[host] >= a.host_cap:
                    why = "host_cap"
                else:
                    k = exact_key(text)
                    if k in exact:
                        why = "exact_dup"
                    elif near.seen(text):
                        why = "near_dup"
                    else:
                        exact.add(k)
            reasons[why or "kept"] += 1
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
        with open(os.path.join(a.out, "build_corpus_stats.json"), "w") as f:
            json.dump(
                {"reasons": reasons, "top_hosts": hosts.most_common(50)}, f, ensure_ascii=False, indent=1
            )


if __name__ == "__main__":
    main()
