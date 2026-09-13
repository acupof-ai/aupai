#!/usr/bin/env python3
"""Stage a vetted textbook corpus into a train-readable shard domain.

Vetted textbooks land as `claude-gen-A.jsonl` / `claude-gen-B.jsonl` with a `text`
key. train._domain_seqs only globs shard names (SHARD_RE `_NNNN.jsonl`) and reads the
`content` key, so a vetted dir cannot be tokenized directly. This script converts a
COPY into a new domain: one `<dest>_NNNN.jsonl` shard per vetted row file, content key
only, and a build_corpus_stats.json carrying the live fingerprint plus the vet run's
decontam fingerprint.

Writes each shard to a temp file then os.replace; never links the source (a hardlink
opened for writing truncates the vetted original, gate_failure_incidents.md §302).

Usage:
  python scripts/stage_textbook_domain.py \
      --src data/corpus/textbooks_claude_v41_vetted --dest textbooks_claude_v41_train
  python scripts/stage_textbook_domain.py --selftest
"""
import argparse
import hashlib
import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import train  # noqa: E402

VET_STATS = "vet_textbooks_stats.json"
VET_FAILURES = {"vet_failures.jsonl"}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 26), b""):
            h.update(blk)
    return h.hexdigest()


def vetted_row_files(src):
    files = []
    for nm in sorted(os.listdir(src)):
        if not nm.endswith(".jsonl") or nm in VET_FAILURES:
            continue
        files.append(os.path.join(src, nm))
    if not files:
        raise SystemExit(f"no vetted *jsonl row files (excluding {sorted(VET_FAILURES)}) in {src}")
    return files


def stage(src, dst, force=False):
    src = os.path.abspath(src)
    dst = os.path.abspath(dst)
    if not os.path.isdir(src):
        raise SystemExit(f"src missing: {src}")
    stats_p = os.path.join(src, VET_STATS)
    if not os.path.exists(stats_p):
        raise SystemExit(f"missing {stats_p}: a stage without the vet stats cannot carry decon_fp")
    with open(stats_p, encoding="utf-8") as fh:
        vet = json.load(fh)
    decon_fp = None
    for k in ("decontam_fp", "decon_fp"):
        if vet.get(k):
            decon_fp = vet[k]
            break
    if not decon_fp:
        raise SystemExit(f"{VET_STATS} carries no decontam_fp/decon_fp; refusing an unstamped stage")

    files = vetted_row_files(src)
    os.makedirs(dst, exist_ok=True)
    existing = {n for n in os.listdir(dst) if n.endswith(".jsonl")}
    if existing and not force:
        raise SystemExit(f"{dst} holds {len(existing)} jsonl files; pass --force to re-stage")

    prefix = os.path.basename(dst)
    total = 0
    sources = []
    for i, src_file in enumerate(files):
        nm = f"{prefix}_{i:04d}.jsonl"
        out_p = os.path.join(dst, nm)
        n = 0
        tmp_p = out_p + ".tmp"
        with open(tmp_p, "w", encoding="utf-8") as out, \
                open(src_file, encoding="utf-8") as fh:
                for ln in fh:
                    if not ln.strip():
                        continue
                    r = json.loads(ln)
                    if "text" not in r:
                        raise SystemExit(f"{src_file} row {n}: no 'text' key")
                    out.write(json.dumps({"content": r["text"]}, ensure_ascii=False) + "\n")
                    n += 1
        os.replace(tmp_p, out_p)
        if os.stat(out_p).st_ino == os.stat(src_file).st_ino:
            raise SystemExit(f"inode collision with source {src_file}; refusing to continue")
        total += n
        sources.append({"file": os.path.basename(src_file), "rows": n, "sha256": sha256_file(src_file)})

    live_fp = train._corpus_fp(dst)
    stats = {
        "domain": os.path.basename(dst),
        "fingerprint": live_fp,
        "source_vetted_dir": os.path.basename(src),
        "vet_decon_fp": decon_fp,
        "vet_tokens_out": vet.get("tokens_out"),
        "vet_rows_out": vet.get("rows_out"),
        "vet_exec_pass_rate": vet.get("exec_pass_rate"),
        "staged_rows": total,
        "staged_shards": len(files),
        "sources": sources,
        "note": "train-readable COPY of vetted textbooks (text->content, shard names); "
                "never links the source",
    }
    with open(os.path.join(dst, "build_corpus_stats.json.tmp"), "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=1)
    os.replace(os.path.join(dst, "build_corpus_stats.json.tmp"),
               os.path.join(dst, "build_corpus_stats.json"))
    print(f"STAGED {os.path.basename(dst)} shards={len(files)} rows={total} fp={live_fp} "
          f"vet_decon_fp={decon_fp}")
    return stats


def _selftest():
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "vetted")
        os.makedirs(src)
        with open(os.path.join(src, "claude-gen-A.jsonl"), "w", encoding="utf-8") as fh:
            for t in ("print('a')\nassert True", "x = 1\nassert x == 1"):
                fh.write(json.dumps({"text": t, "n": len(t) // 4}) + "\n")
        with open(os.path.join(src, "vet_failures.jsonl"), "w") as fh:
            fh.write('{"bad":1}\n')
        with open(os.path.join(src, VET_STATS), "w") as fh:
            json.dump({"tokens_out": 42, "rows_out": 2, "exec_pass_rate": 0.99,
                       "decontam_fp": "0aefe6a2aa5e130f"}, fh)
        dst = os.path.join(d, "vetted_train")
        st = stage(src, dst)
        names = sorted(os.listdir(dst))
        assert names == ["build_corpus_stats.json", "vetted_train_0000.jsonl"], names
        assert st["staged_rows"] == 2 and st["vet_decon_fp"] == "0aefe6a2aa5e130f"
        with open(os.path.join(dst, "vetted_train_0000.jsonl"), encoding="utf-8") as fh:
            row = json.loads(fh.readline())
        assert set(row) == {"content"} and row["content"].startswith("print")
        assert train._corpus_fp(dst) == st["fingerprint"]
        try:
            stage(src, dst)
        except SystemExit:
            pass
        else:
            raise AssertionError("restage without --force must refuse")
        stage(src, dst, force=True)
        print("selftest ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(ROOT, "data", "corpus",
                                                   "textbooks_claude_v41_vetted"))
    ap.add_argument("--dest", default=os.path.join(ROOT, "data", "corpus",
                                                    "textbooks_claude_v41_train"))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
        return
    stage(a.src, a.dest, a.force)


if __name__ == "__main__":
    main()
