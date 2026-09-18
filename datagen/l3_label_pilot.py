"""Orchestrate L3 multi-dimensional teacher labels over a stratified pilot.

# restartable: resumable by sample_id — completed doc ids are read from --out on start
# and skipped, labels/rejections are appended per row, so an interrupt re-sends only
# the unfinished tail; the ledger double-write is a post-pass append over this run's rows.

CPU orchestration, but the teacher is a remote OpenAI-compatible server. Backends:
- stub       deterministic offline non-label (plumbing tests).
- openai     one endpoint (--url) OR many (--endpoints a,b,c) round-robin load-balanced
             with a thread pool (IO-bound; --workers total concurrent requests).

Output jsonl per row with rubric_version + teacher_model + stratum metadata; rejects go
to <out>.rejected.jsonl and are counted (never fabricated); resumable by sample_id.
Throughput (doc/s), per-endpoint latency, and retry/error counts print at the end.
"""

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.request

from l3_rubric import RUBRIC_VERSION, build_prompt, parse_scores, select_rubric


def _ts():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def stub_teacher(text: str, rubric: dict) -> str:
    """Deterministic non-label: scores track length bands so the whole pipeline runs
    offline. Explicitly not a judgement; backend must be 'openai' for real labels."""
    n = len(text)
    band = 1 if n < 400 else 2 if n < 1200 else 3 if n < 3000 else 4 if n < 8000 else 5
    # vary one dim off a hash so a parser bug hiding constant scores is visible
    j = int(hashlib.sha256(text[:200].encode()).hexdigest()[:6], 16)
    scores = {
        "content_quality": band,
        "factual_correctness": max(1, min(5, band + (j % 3) - 1)),
        "complexity": max(1, min(5, band - (j % 2))),
        "educational_or_code_value": band,
    }
    return json.dumps(scores)


def _chat_url(url: str) -> str:
    base = url.rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    return base + "/chat/completions"


class OpenAIPool:
    """Round-robin pool of OpenAI-compatible endpoints with a thread pool.

    Each request is handed the NEXT endpoint under a lock (client-side load balance);
    the server replicas are identical, so retrying a different endpoint on failure is
    safe. Tracks per-endpoint call count / total latency for the throughput report.
    """

    def __init__(self, endpoints, model, timeout, max_tokens, retries):
        self.endpoints = [_chat_url(u) for u in endpoints]
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.retries = retries
        self._i = 0
        self._lock = threading.Lock()
        self.stats = {u: {"calls": 0, "latency": 0.0, "errors": 0} for u in endpoints}

    def _next(self) -> str:
        with self._lock:
            u = self.endpoints[self._i % len(self.endpoints)]
            self._i += 1
            return u

    def _one_request(self, endpoint_url, label, prompt) -> str:
        body = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": self.max_tokens,
                # REQUIRED for Qwen3.5: without enable_thinking=false the model spends
                # the budget on a CoT trace and returns no parseable rubric JSON.
                "chat_template_kwargs": {"enable_thinking": False},
            }
        ).encode()
        req = urllib.request.Request(
            endpoint_url, data=body, headers={"Content-Type": "application/json"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            d = json.loads(r.read())
        dt = time.time() - t0
        with self._lock:
            key = next(k for k in self.stats if _chat_url(k) == endpoint_url)
            self.stats[key]["calls"] += 1
            self.stats[key]["latency"] += dt
        _ = label
        return d["choices"][0]["message"]["content"]

    def ask(self, text, rubric):
        prompt = build_prompt(text, rubric)
        last = None
        for attempt in range(self.retries + 1):
            url = self._next()
            try:
                return self._one_request(url, rubric.get("kind"), prompt)
            except (urllib.error.URLError, OSError, KeyError,
                    json.JSONDecodeError, TimeoutError) as e:
                last = e
                with self._lock:
                    key = next(k for k in self.stats if _chat_url(k) == url)
                    self.stats[key]["errors"] += 1
                time.sleep(min(2 ** attempt, 8))
        raise RuntimeError(f"teacher failed after {self.retries + 1} tries: {last}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", required=True, help="stratified sample jsonl")
    ap.add_argument("--out", required=True, help="labels jsonl")
    ap.add_argument("--backend", choices=["stub", "openai"], default="stub")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--endpoints", default=None,
                    help="comma-separated OpenAI-compatible base urls; load-balanced")
    ap.add_argument("--model", default="teacher")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--max-tokens", type=int, default=80)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--workers", type=int, default=1,
                    help="total concurrent teacher requests (IO-bound)")
    ap.add_argument("--limit", type=int, default=None, help="cap rows this run (pilot)")
    ap.add_argument("--ledger", default=None,
                    help="optional frozen score-ledger jsonl to ALSO append rows to")
    ap.add_argument("--domain", default=None,
                    help="full corpus domain name for ledger rows; required with --ledger")
    args = ap.parse_args()

    if args.ledger and not args.domain:
        raise SystemExit("REFUSE: --ledger requires --domain (full corpus domain name)")

    endpoints = ([u.strip() for u in args.endpoints.split(",") if u.strip()]
                 if args.endpoints else [args.url])
    if args.backend == "openai":
        pool = OpenAIPool(endpoints, args.model, args.timeout, args.max_tokens,
                         args.retries)
        teacher, teacher_id = pool.ask, args.model
    else:
        pool, teacher, teacher_id = None, stub_teacher, "stub-length-function"

    done = set()
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as fh:
            for line in fh:
                done.add(json.loads(line)["sample_id"])

    # collect the work: (row, rubric) for every not-done, within --limit
    todo = []
    with open(args.pilot, encoding="utf-8") as inp:
        for line in inp:
            row = json.loads(line)
            if row["sample_id"] in done:
                continue
            if args.limit is not None and len(todo) >= args.limit:
                break
            todo.append((row, select_rubric(row["content"])))

    def _work(item):
        row, rubric = item
        try:
            raw = teacher(row["content"], rubric)
            scores = parse_scores(raw, rubric)
            return ("ok", row, rubric, scores)
        except Exception as e:  # loud: persisted, counted, never fabricated
            return ("rej", row, rubric, str(e))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    rej_path = args.out + ".rejected.jsonl"
    kept = rej = 0
    ledger_rows = []
    wall0 = time.time()
    # concurrent map preserves submission order on the executor; gather, then write in
    # order so the output is deterministic and resumable.
    with open(args.out, "a", encoding="utf-8") as out, \
            open(rej_path, "a", encoding="utf-8") as rej_fh, \
            cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for status, row, rubric, payload in ex.map(_work, todo):
            sid = row["sample_id"]
            if status == "rej":
                rej_fh.write(json.dumps(
                    {"sample_id": sid, "reason": payload,
                     "rubric_kind": rubric["kind"], "ts": _ts()},
                    ensure_ascii=False) + "\n")
                rej += 1
                continue
            scores = payload
            ts = _ts()
            label_id = hashlib.sha256((sid + teacher_id).encode()).hexdigest()[:16]
            out.write(json.dumps({
                "label_id": label_id,
                "sample_id": sid,
                "language": row.get("language"),
                "length_band": row.get("length_band"),
                "source": row.get("source"),
                "url": row.get("url"),
                "rubric_version": RUBRIC_VERSION,
                "rubric_kind": rubric["kind"],
                "teacher_model": teacher_id,
                "backend": args.backend,
                "scores": scores,
                "stratum": {"language": row.get("language"),
                            "length_band": row.get("length_band")},
                "ts": ts,
            }, ensure_ascii=False) + "\n")
            if args.ledger:
                ledger_rows.append({
                    "doc_id": sid, "domain": args.domain,
                    "lang": str(row.get("language") or "_"),
                    "scorer_name": "l3-rubric", "scorer_version": RUBRIC_VERSION,
                    "ts": ts, "score": None, "rubric_dims": scores, "cut": None,
                    "model": teacher_id, "backend": args.backend,
                    "stratum": {"language": row.get("language"),
                                "length_band": row.get("length_band")},
                    "rubric_kind": rubric["kind"], "record_id": label_id,
                    "src_sha": None,
                })
            kept += 1

    wall = max(1e-6, time.time() - wall0)
    ledger_n = 0
    if args.ledger and ledger_rows:
        from datagen.score_ledger import append_rows  # clean dependency on #389/main

        ledger_n = append_rows(args.ledger, ledger_rows)

    report = {
        "pilot": args.pilot, "out": args.out, "backend": args.backend,
        "endpoints": endpoints if args.backend == "openai" else 1,
        "workers": args.workers, "teacher_model": teacher_id,
        "rubric_version": RUBRIC_VERSION,
        "attempted": len(todo), "kept": kept, "rejected": rej,
        "success_rate": round(kept / max(1, kept + rej), 4),
        "wall_seconds": round(wall, 1), "doc_per_second": round(kept / wall, 2),
        "rejected_path": rej_path,
        "ledger": args.ledger, "ledger_domain": args.domain,
        "ledger_appended": ledger_n, "ts": _ts(),
    }
    if pool is not None:
        report["endpoint_stats"] = {
            u: {"calls": s["calls"], "errors": s["errors"],
                "avg_latency_s": round(s["latency"] / max(1, s["calls"]), 3)}
            for u, s in pool.stats.items()}
    with open(args.out + ".label_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
