#!/usr/bin/env python3
"""Build the phase-2 textbook domain from the delivered claude textbooks.

Reads data/corpus/textbooks_claude_v41/gen*.jsonl, then in order:
  1. skip every file named in the source manifest.json under any anneal_excluded*
     key (old-prompt gen-A and sub-1800-word gen-B rows);
  2. optional --keep-list <jsonl>: restrict to rows whose (seed_topic, lens) is in
     the list (3b mutation-gate result). Re-running with a keep-list rebuilds;
  3. dedup: among rows with the same (seed_topic, lens) keep the longest text;
  4. 13-gram decontaminate the row `text` against HumanEval+MBPP via
     filters/decontam_ngram.py, dropping contaminated rows whole.

Writes data/corpus/textbook_claude_v41_dc/ as one sharded jsonl plus
build_corpus_stats.json with the three-way counts (scanned pre-dedup / kept
post-dedup / kept post-decontam). Source and existing mix domains are never
written. Restartable: the shard is rewritten whole each run.

    python3 scripts/build_textbook_domain.py
    python3 scripts/build_textbook_domain.py --keep-list /path/mutation_kept.jsonl
"""
# restartable: an interrupt costs only the single ~seconds full-domain rebuild; every
# run rewrites the output shard and stats whole from immutable inputs, so re-running
# converges with no partial-state recovery.
import argparse
import glob
import json
import os
import sys

ROOT = os.environ.get("AUPAI_ROOT",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

SRC = os.path.join("data", "corpus", "textbooks_claude_v41")
DST = "textbook_claude_v41_dc"
N = 13


def excluded_files(src_dir):
    p = os.path.join(src_dir, "manifest.json")
    out = set()
    if not os.path.exists(p):
        return out
    with open(p, encoding="utf-8") as fh:
        m = json.load(fh)
    for k, v in m.items():
        if k.startswith("anneal_excluded") and isinstance(v, dict):
            out.update(v.get("files", []))
    return out


def key_of(r):
    return (r.get("seed_topic"), r.get("lens"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--src", default="")
    ap.add_argument("--keep-list", default="",
                    help="optional jsonl; restrict to its (seed_topic,lens) keys")
    ap.add_argument("--out-name", default=DST)
    a = ap.parse_args()
    root = a.root
    src_dir = a.src or os.path.join(root, SRC)
    dst_dir = os.path.join(root, "data", "corpus", a.out_name)
    os.makedirs(dst_dir, exist_ok=True)

    keep_keys = None
    if a.keep_list:
        keep_keys = set()
        with open(a.keep_list, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    keep_keys.add(key_of(r))

    excluded = excluded_files(src_dir)
    files = sorted(p for p in glob.glob(os.path.join(src_dir, "gen*.jsonl"))
                   if os.path.basename(p) not in excluded)

    from filters.decontam_ngram import Decontaminator, decontam_fp
    decon = Decontaminator.load_default(root)

    scanned = 0
    excluded_by_keep = 0
    best = {}  # (seed_topic,lens) -> row (longest text)
    per_file_in = {}
    for p in files:
        n = 0
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                scanned += 1
                n += 1
                if keep_keys is not None and key_of(r) not in keep_keys:
                    excluded_by_keep += 1
                    continue
                k = key_of(r)
                if k not in best or len(r.get("text", "")) > len(best[k].get("text", "")):
                    best[k] = r
        per_file_in[os.path.basename(p)] = n

    dedup_kept = len(best)
    dedup_dropped = scanned - excluded_by_keep - dedup_kept

    # decontaminate the dedup winners, grouped back per source file name (recorded on row?
    # rows do not carry file; emit one shard per seed_topic bucket is unnecessary -- write a
    # single shard, matching how mix readers glob *.jsonl regardless of shard layout).
    clean = []
    decon_dropped = 0
    problems = {}
    for r in best.values():
        h = decon.hit(r.get("text", ""))
        if h:
            decon_dropped += 1
            problems[h["problem"]] = problems.get(h["problem"], 0) + 1
        else:
            clean.append(r)
    final = dedup_kept - decon_dropped

    out_shard = os.path.join(dst_dir, "textbook_claude_v41_dc_000.jsonl")
    with open(out_shard, "w", encoding="utf-8") as fh:
        for r in clean:
            # train._jsonl_content reads the "content" key; textbook rows carry "text".
            # Project to the training schema, keeping provenance fields alongside.
            out = dict(r)
            out["content"] = r.get("text", "")
            fh.write(json.dumps(out, ensure_ascii=False) + "\n")

    from datagen.corpus_fingerprint import fp_dir
    total_n = sum(int(r.get("n") or 0) for r in clean)
    stats = {
        "domain": a.out_name,
        "source_domain": os.path.basename(src_dir),
        "filter": "filters/decontam_ngram.py 13-word-token containment vs HumanEval+MBPP",
        "n": N,
        "files_read": len(files),
        "files_excluded_manifest": len(excluded),
        "rows_scanned_pre_dedup": scanned,
        "rows_removed_by_keep_list": excluded_by_keep,
        "rows_after_dedup": dedup_kept,
        "dedup_collapsed": dedup_dropped,
        "rows_decontam_dropped": decon_dropped,
        "rows_final": final,
        "sum_n_field_final": total_n,
        "distinct_problems_hit": len(problems),
        "problems": problems,
        "decontam_fp": decontam_fp(
            os.path.join(root, "data", "eval", "humaneval", "humaneval_164.jsonl"),
            os.path.join(root, "data", "eval", "mbpp_holdouts.jsonl")),
        "fingerprint": fp_dir(dst_dir),
    }
    with open(os.path.join(dst_dir, "build_corpus_stats.json"), "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=1)

    print(f"files {len(files)} read, {len(excluded)} manifest-excluded")
    print(f"scanned={scanned} keep_list_removed={excluded_by_keep} "
          f"after_dedup={dedup_kept} (collapsed {dedup_dropped}) "
          f"decontam_dropped={decon_dropped} FINAL={final} sum_n={total_n}")
    print(f"wrote {out_shard}")


if __name__ == "__main__":
    main()
