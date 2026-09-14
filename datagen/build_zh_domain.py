#!/usr/bin/env python3
"""Build the 0e-6 Chinese gate-vocab domain from wikipedia-cn-20230720-filtered.

Pipeline (uniform with the gate): JSON {completion} docs -> nonempty/length floor
-> secrets redaction -> 13-gram HumanEval/MBPP decontam (filters.decontam_ngram;
kept for uniformity though Chinese prose rarely matches code prompts) -> exact
normalized dedup -> exact per-doc new-vocab token count, stopping at the target.

Output data/corpus/<domain>/{prefix}_NNN.jsonl with {content,source,url} and a
build_corpus_stats.json carrying the canonical fp_dir fingerprint and
decontam_fp. The 13-gram AGGREGATE pass is not re-run here because there is no
per-group stage: this converter calls decon() per doc once, exactly like the
L2/L3 per-shard substring stage; for a single-source single-pass domain the
cross-group dedup and the aggregate coincide.
"""
import argparse
import glob
import hashlib
import json
import os
import sys

import zstandard

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenizers import Tokenizer  # noqa: E402

from datagen.corpus_fingerprint import fp_dir  # noqa: E402
from datagen.gen_exercises import _norm  # noqa: E402
from filters.decontam_ngram import Decontaminator, decontam_fp  # noqa: E402
from filters.secrets import redact_text  # noqa: E402

SHARD = 100 * 1024 * 1024
MIN_CHARS = 200


class Writer:
    def __init__(self, d, prefix):
        os.makedirs(d, exist_ok=True)
        self.d, self.prefix, self.n, self.fh, self.b = d, prefix, 0, None, 0

    def write(self, rec):
        line = json.dumps(rec, ensure_ascii=False) + "\n"
        size = len(line.encode())
        if self.fh is None or self.b + size > SHARD:
            if self.fh:
                self.fh.close()
            self.fh = open(os.path.join(self.d, f"{self.prefix}_{self.n:03d}.jsonl"), "w")
            self.n += 1
            self.b = 0
        self.fh.write(line)
        self.b += size

    def close(self):
        if self.fh:
            self.fh.close()


def iter_docs(src, content_key):
    if src.endswith(".zst"):
        import io as _io

        d = zstandard.ZstdDecompressor()
        with d.stream_reader(open(src, "rb")) as r, _io.TextIOWrapper(r, encoding="utf-8") as t:
            for line in t:
                yield json.loads(line)
    elif src.endswith(".jsonl"):
        with open(src, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
    else:
        yield from json.load(open(src))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, action="append",
                    help="input file; repeatable, or a glob (.zst/.jsonl streamed, else JSON array)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--content-key", default="completion")
    ap.add_argument("--tokenizer", default="data/tokenizer.json")
    ap.add_argument("--target-tokens", type=int, default=800_000_000)
    ap.add_argument("--source", default="pleisto/wikipedia-cn-20230720-filtered")
    args = ap.parse_args()

    srcs = []
    for s in args.src:
        srcs.extend(sorted(glob.glob(s)) if "*" in s or "?" in s else [s])
    tok = Tokenizer.from_file(args.tokenizer)
    decon = Decontaminator.load_default(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    domain = os.path.basename(args.out.rstrip("/"))
    w = Writer(args.out, domain)
    seen = set()
    kept = kept_chars = kept_tokens = scanned = 0
    reasons = {"empty_short": 0, "secret_redacted": 0, "decontam": 0, "dup": 0}
    stop = False

    for src in srcs:
        if stop:
            break
        for d in iter_docs(src, args.content_key):
            scanned += 1
            c = (d.get(args.content_key) or "").strip()
            if len(c) < MIN_CHARS:
                reasons["empty_short"] += 1
                continue
            c, nsec = redact_text(c)
            reasons["secret_redacted"] += 1 if nsec else 0
            if decon.hit(c) is not None:
                reasons["decontam"] += 1
                continue
            sig = hashlib.sha1(_norm(c).encode()).hexdigest()
            if sig in seen:
                reasons["dup"] += 1
                continue
            seen.add(sig)
            nt = len(tok.encode(c).ids) + 1
            if kept_tokens + nt > args.target_tokens and kept:
                stop = True
                break
            w.write({"content": c, "source": args.source, "url": d.get("url") or ""})
            kept += 1
            kept_chars += len(c)
            kept_tokens += nt
    w.close()

    n_shards = len(glob.glob(os.path.join(args.out, f"{domain}_[0-9]*.jsonl")))
    ngram_fp = decontam_fp(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "data", "eval", "humaneval", "humaneval_164.jsonl"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "data", "eval", "mbpp_holdouts.jsonl"))
    canonical = fp_dir(args.out)
    record = {
        "domain": domain,
        "source": args.source,
        "input_files": srcs,
        "docs_scanned": scanned,
        "kept": kept,
        "kept_chars": kept_chars,
        "kept_tokens": kept_tokens,
        "tokens": kept_tokens,
        "tokens_status": "measured",
        "tokens_config": f"{args.tokenizer}, exact per-doc ids+1",
        "filters": "min200chars+secrets-redact+13gram-decontam(humaneval,mbpp)+exact-dedup",
        "n_shards": n_shards,
        "decontam_fp": ngram_fp,
        "fingerprint": canonical,
        "total_rows": scanned,
        "reasons": reasons,
        "target_tokens": args.target_tokens,
        "target_reached": kept_tokens >= args.target_tokens,
    }
    json.dump(record, open(os.path.join(args.out, "build_corpus_stats.json"), "w"), indent=1)
    print(f"ZH_KEPT docs={kept} tokens={kept_tokens} ({kept_tokens/1e9:.3f}B) "
          f"chars={kept_chars} shards={n_shards} reasons={reasons} fp={canonical}", flush=True)


if __name__ == "__main__":
    main()
