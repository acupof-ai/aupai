#!/usr/bin/env python3
"""Report-only credential scan over the raw corpus shards. NO DELETION, NO FILTER CHANGE.

User order via 4c, 2026-09-05, step 1: after a live third-party Postgres credential reached a
public commit through data/probes/api_cloze.jsonl, find out whether the TRAINING CORPUS carries
the same class of content. The corpus is 81 GB of scraped public code, so the prior is that it
does -- the question is how much and in what shapes, because that decides whether a filter is
worth a rebuild of every affected shard (filters_fp changes, so it is not free).

THIS SCRIPT DELETES NOTHING AND CHANGES NO FILTER. It writes counts.

VALUES ARE NEVER PRINTED. Per pattern per domain: a count, and the row ids that hold a match.
A report that quoted the matches would take credentials out of a 81 GB corpus nobody reads and
concentrate them into a small file, a terminal, a log and a session transcript -- strictly worse
than leaving them where they are. The row id is enough to act on later.

ONE PROCESS, CPU, per 4c. It reads shards off /data00, which is the co-residency hazard
AGENTS.md names (an eval that reads a token cache waits for the run) -- but these are the RAW
JSONL shards, not token caches, and the read is sequential and streamed a line at a time rather
than torch.load of a whole cache. Still: it takes --max-bytes-per-domain so a first pass can be
bounded, and it prints throughput so the full-pass cost is measured rather than guessed.
"""

import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from credential_scan import PATTERNS, EXAMPLES  # noqa: E402


def scan_line(line):
    """[(pattern name)] for one raw line. Never returns the matched text."""
    for ex in EXAMPLES:
        if ex in line:
            line = line.replace(ex, "<published-example>")
    return [name for name, pat, _desc in PATTERNS if pat.search(line)]


def row_id(obj, n):
    """A stable handle for a matching row that is not its content."""
    for k in ("id", "doc_id", "url", "path", "repo_name", "file_path"):
                                    # a URL or path is provenance, not the secret; but it can
                                    # itself carry a credential (that is the db-uri shape), so
                                    # it is truncated and only used when short and clean.
        v = obj.get(k)
        if isinstance(v, str) and v and not scan_line(v):
            return f"{k}={v[:80]}"
    return f"line={n}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", nargs="+", required=True)
    ap.add_argument("--root", default="/work/aupai")
    ap.add_argument("--max-bytes-per-domain", type=float, default=0,
                    help="0 = whole domain; otherwise stop after roughly this many bytes")
    ap.add_argument("--max-ids", type=int, default=40,
                    help="row ids recorded per (domain, pattern); counts are always complete")
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip-done", action="store_true",
                    help="read --out back and skip domains already recorded complete there")
    a = ap.parse_args()

    report = {"scanned": {}, "findings": {}, "throughput": {}}

    # WRITES PER DOMAIN, not once at the end. `harness check restartability` flagged the
    # end-only version and it was right: 81 GB across six code domains at the measured rate is
    # a long single pass, and a version that accumulates in memory and writes at exit loses
    # every completed domain to one interrupt -- the pod's own history is full of jobs killed
    # by a dropped tunnel or a card reclaim. Each domain's result lands as soon as it is known,
    # and --skip-done reads the partial file back so a re-run continues instead of restarting.
    def flush():
        if not a.out:
            return
        tmp = a.out + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1, ensure_ascii=False)
        os.replace(tmp, a.out)   # atomic: a reader never sees a half-written report

    done = set()
    if a.out and a.skip_done and os.path.exists(a.out):
        try:
            with open(a.out, encoding="utf-8") as f:
                prev = json.load(f)
            report["scanned"].update(prev.get("scanned", {}))
            report["findings"].update(prev.get("findings", {}))
            report["throughput"].update(prev.get("throughput", {}))
            done = {k for k, v in prev.get("scanned", {}).items() if not v.get("partial")}
            print(f"resuming: {len(done)} domain(s) already complete in {a.out}", flush=True)
        except (OSError, ValueError) as e:
            print(f"could not read {a.out} to resume ({e}); scanning everything", flush=True)

    t_all = time.time()
    for dom in a.domains:
        if dom in done:
            print(f"{dom}: already complete, skipping (--skip-done)", flush=True)
            continue
        d = os.path.join(a.root, "data", "corpus", dom)
        shards = sorted(glob.glob(os.path.join(d, "*.jsonl")))
        if not shards:
            print(f"{dom}: no .jsonl shards", flush=True)
            continue
        t0 = time.time()
        n_rows = n_bytes = 0
        hits = {}
        ids = {}
        stop = False
        for sh in shards:
            if stop:
                break
            with open(sh, encoding="utf-8", errors="replace") as f:
                for i, line in enumerate(f, 1):
                    n_rows += 1
                    n_bytes += len(line)
                    for name in scan_line(line):
                        hits[name] = hits.get(name, 0) + 1
                        if len(ids.setdefault(name, [])) < a.max_ids:
                            try:
                                obj = json.loads(line)
                            except ValueError:
                                obj = {}
                            ids[name].append(f"{os.path.basename(sh)}:{row_id(obj, i)}")
                    if a.max_bytes_per_domain and n_bytes >= a.max_bytes_per_domain:
                        stop = True
                        break
        el = time.time() - t0
        rate_mb = (n_bytes / 1e6) / el if el else 0
        report["scanned"][dom] = {"rows": n_rows, "bytes": n_bytes, "shards": len(shards),
                                  "seconds": round(el, 1), "MB_per_s": round(rate_mb, 1),
                                  "partial": bool(a.max_bytes_per_domain and stop)}
        report["findings"][dom] = {"counts": hits, "row_ids": ids}
        report["throughput"][dom] = round(rate_mb, 1)
        pretty = ", ".join(f"{k}={v}" for k, v in sorted(hits.items())) or "none"
        print(f"{dom}: {n_rows:,} rows, {n_bytes/1e9:.2f} GB, {el:.0f}s, {rate_mb:.0f} MB/s "
              f"-> {pretty}", flush=True)
        flush()   # this domain's result survives an interrupt from here on

    tot_bytes = sum(v["bytes"] for v in report["scanned"].values())
    tot_rows = sum(v["rows"] for v in report["scanned"].values())
    el_all = time.time() - t_all
    report["total"] = {"rows": tot_rows, "bytes": tot_bytes, "seconds": round(el_all, 1),
                       "MB_per_s": round((tot_bytes / 1e6) / el_all, 1) if el_all else 0}
    # THE detect-secrets COST, at the rate its own docstring records rather than a guess:
    # scripts/build_agentic_sft.py:157 measures ~2 ms/line (500 lines in 0.99 s) with chunk=1,
    # which it needs because scan_file under-reports on multi-line documents.
    ds_hours = tot_rows * 0.002 / 3600
    report["detect_secrets_estimate"] = {
        "ms_per_line": 2, "basis": "build_agentic_sft.find_secrets docstring, 500 lines in 0.99s",
        "rows": tot_rows, "single_process_hours": round(ds_hours, 1),
        "hours_at_180_cores": round(ds_hours / 180, 2)}
    print(f"\ntotal: {tot_rows:,} rows, {tot_bytes/1e9:.2f} GB in {el_all/60:.1f} min "
          f"({report['total']['MB_per_s']:.0f} MB/s)")
    print(f"detect-secrets at 2 ms/line: {ds_hours:,.0f} h single-process, "
          f"{ds_hours/180:.1f} h across the pod's 180 cores (ESTIMATE from a recorded rate)")
    if a.out:
        flush()
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
