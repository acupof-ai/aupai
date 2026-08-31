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
    with op(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                text = d.get("content") or d.get("text")
                if not text and d.get("instruction"):
                    # Same ChatML the SFT pack uses, so pretraining already sees that format.
                    text = "".join(format_example(d["instruction"], d.get("output", "")))
                yield text or "", d.get("url")


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", action="append", required=True, help="fineweb2 | skypile | jsonl:<glob>")
    ap.add_argument("--target_tokens", type=float, default=8e9)
    ap.add_argument("--domain", required=True, help="a default-mix domain -> data/corpus/<domain>/")
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
    a = ap.parse_args()
    a.out = a.out or os.path.join(OUT_DIR, a.domain)

    target_chars = a.target_tokens * CHARS_PER_TOKEN
    reasons, hosts = Counter(), Counter()
    exact, near = set(), MinHashLSH()
    reject = reject_light if a.filters == "light" else reject_reason
    # (a) refuse at START: a code-family / English-leaning domain run with the
    # web chain rejects ~100% as not_zh after processing millions of docs (2026-
    # 08-31, twice, ~1h each) -- the final 0-kept guard is too late. Keyed on
    # the explicit domain names; a rename creates a new domain and must rejoin.
    CODE_DOMAINS = {"code", "code_rp1t", "en", "math", "cot"}
    if a.domain in CODE_DOMAINS and a.filters != "light":
        raise SystemExit(
            f"REFUSE: domain '{a.domain}' is code-family / English-leaning; --filters must be "
            f"'light' (the web CJK/digit/symbol chain rejects it ~100%). Got --filters "
            f"'{a.filters}'."
        )
    for pat in a.exclude:
        n0 = len(exact)
        for path in sorted(glob.glob(pat)):
            for text, _ in iter_jsonl(path):
                exact.add(exact_key(text))
        print(f"exclude {pat}: dedup set +{len(exact) - n0} docs", flush=True)
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
