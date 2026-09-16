#!/usr/bin/env python3
"""Drive the L2 teacher over a pool across multiple sglang endpoints concurrently.

l3_label_pilot.py is the single-endpoint reference path; this is the throughput path
for labeling the 50k L2 pool. It reuses l3_rubric (select/build/parse) and writes the
SAME label row schema, plus #389 score-ledger rows (rubric_dims, XOR with score).

Each worker is pinned to one endpoint (round-robin at submit) so per-endpoint JSON
success and throughput are measured separately -- six cards do not combine linearly,
so the aggregate is measured at --limit 100 before the full run, never assumed.

Outputs (all append; the run resumes by sample_id):
  --out      label rows, one JSON per line (l3 pilot schema)
  <out>.rejected.jsonl   one line per parse/HTTP failure, with endpoint + reason
  --ledger   optional frozen score ledger; appended in validated batches

# restartable: completed sample_ids in --out are skipped; ledger rows append in batches.
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import sys
import threading
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from l3_rubric import RUBRIC_VERSION, build_prompt, parse_scores, select_rubric  # noqa: E402

KIND_DOMAIN = {"nl": "en_c4_stage2_dc", "code": "code_py_starcoder_dc"}


def _post(url, model, timeout, text, rubric):
    base = url.rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": build_prompt(text, rubric)}],
        "temperature": 0.0,
        "max_tokens": 200,
        # REQUIRED for Qwen3.x: without this the budget goes to CoT and no rubric JSON.
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(base + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


def _label_one(item, endpoints, model, timeout):
    """item=(idx, row); returns (ok, payload, ep). One retry on a transient HTTP error."""
    idx, row = item
    ep = endpoints[idx % len(endpoints)]
    text = row["content"]
    rubric = select_rubric(text)
    last = None
    for _ in range(2):
        try:
            raw = _post(ep, model, timeout, text, rubric)
            scores = parse_scores(raw, rubric)
            return True, (row, scores, rubric["kind"], ep), ep
        except Exception as e:  # loud: persist the failure, never fabricate a label
            last = f"{type(e).__name__}: {e}"
    return False, (row.get("sample_id"), last, rubric["kind"], ep), ep


def _label_all(todo, out_fh, rej_fh, endpoints, model, ledger, concurrency, timeout):
    lock = threading.Lock()
    stats = {ep: {"ok": 0, "rej": 0} for ep in endpoints}
    ledger_buf = []
    kept = rej = 0
    t0 = time.time()

    def flush_ledger():
        if not (ledger and ledger_buf):
            return
        from score_ledger import append_rows
        n = append_rows(ledger, ledger_buf)
        if n != len(ledger_buf):
            raise SystemExit(f"REFUSE: ledger accepted {n}/{len(ledger_buf)} rows")
        ledger_buf.clear()

    with cf.ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [ex.submit(_label_one, it, endpoints, model, timeout) for it in todo]
        for fu in cf.as_completed(futs):
            ok, payload, ep = fu.result()
            with lock:
                if ok:
                    row, scores, kind, _ = payload
                    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                    label_id = hashlib.sha256(
                        (row["sample_id"] + model).encode()).hexdigest()[:16]
                    out_fh.write(json.dumps({
                        "label_id": label_id,
                        "sample_id": row["sample_id"],
                        "language": row.get("language"),
                        "length_band": row.get("length_band"),
                        "source": row.get("source"),
                        "url": row.get("url"),
                        "rubric_version": RUBRIC_VERSION,
                        "rubric_kind": kind,
                        "teacher_model": model,
                        "backend": "openai",
                        "scores": scores,
                        "stratum": {"language": row.get("language"),
                                    "length_band": row.get("length_band")},
                        "ts": ts,
                    }, ensure_ascii=False) + "\n")
                    if ledger:
                        ledger_buf.append({
                            "doc_id": row["sample_id"],
                            "domain": KIND_DOMAIN.get(row.get("kind"), "_"),
                            "lang": str(row.get("language") or "_"),
                            "scorer_name": "l3-rubric",
                            "scorer_version": RUBRIC_VERSION,
                            "ts": ts,
                            "score": None,
                            "rubric_dims": scores,
                            "cut": None,
                            "model": model,
                            "backend": "openai",
                            "stratum": {"language": row.get("language"),
                                        "length_band": row.get("length_band")},
                            "rubric_kind": kind,
                            "record_id": label_id,
                            "src_sha": None,
                        })
                        if len(ledger_buf) >= 200:
                            flush_ledger()
                    kept += 1
                    stats[ep]["ok"] += 1
                else:
                    sid, reason, kind, _ = payload
                    rej_fh.write(json.dumps({
                        "sample_id": sid, "endpoint": ep,
                        "reason": reason, "rubric_kind": kind,
                        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    }, ensure_ascii=False) + "\n")
                    rej += 1
                    stats[ep]["rej"] += 1
    flush_ledger()
    return kept, rej, stats, time.time() - t0


def drive(pool, out, endpoints, model, *, concurrency, timeout, limit, ledger):
    with open(pool, encoding="utf-8") as fh:
        rows = [json.loads(l) for l in fh]
    done = set()
    if os.path.exists(out):
        with open(out, encoding="utf-8") as fh:
            for line in fh:
                done.add(json.loads(line)["sample_id"])
    todo = [(i, r) for i, r in enumerate(rows) if r["sample_id"] not in done]
    if limit is not None:
        todo = todo[:limit]

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    rej_path = out + ".rejected.jsonl"
    with (
        open(out, "a", encoding="utf-8") as out_fh,
        open(rej_path, "a", encoding="utf-8") as rej_fh,
    ):
        result = _label_all(todo, out_fh, rej_fh, endpoints, model, ledger,
                            concurrency, timeout)
    kept, rej, stats, dt = result
    rate = (kept + rej) / dt if dt else 0.0
    print(json.dumps({
        "pool": pool, "out": out, "attempted": len(todo),
        "kept": kept, "rejected": rej,
        "json_success": round(kept / (kept + rej), 4) if kept + rej else None,
        "elapsed_s": round(dt, 1), "aggregate_doc_per_s": round(rate, 2),
        "per_endpoint": {ep: {"ok": s["ok"], "rej": s["rej"]} for ep, s in stats.items()},
        "concurrency": concurrency,
    }, indent=2))
    return kept, rej, stats


def _selftest() -> int:
    import http.server
    import socketserver
    import tempfile

    from l3_rubric import select_rubric  # noqa: E402

    state = {"flaky_calls": 0}

    class Good(socketserver.TCPServer):
        allow_reuse_address = True

    class Flaky(socketserver.TCPServer):
        allow_reuse_address = True

    class GoodHandler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("content-length", 0)))
            body = json.dumps({"choices": [{"message": {"content": json.dumps({
                "content_quality": 4, "factual_correctness": 3,
                "complexity": 2, "educational_or_code_value": 5})}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    class FlakyHandler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("content-length", 0)))
            state["flaky_calls"] += 1
            if state["flaky_calls"] == 1:
                self.send_response(500)
                self.end_headers()  # fail first attempt -> one retry must recover
            else:
                body = json.dumps({"choices": [{"message": {"content": json.dumps({
                    "content_quality": 1, "factual_correctness": 1,
                    "complexity": 1, "educational_or_code_value": 1})}}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)

        def log_message(self, *a):
            pass

    g = Good(("127.0.0.1", 0), GoodHandler)
    f = Flaky(("127.0.0.1", 0), FlakyHandler)
    import threading
    threading.Thread(target=g.serve_forever, daemon=True).start()
    threading.Thread(target=f.serve_forever, daemon=True).start()
    gu, fu = f"http://127.0.0.1:{g.server_address[1]}", f"http://127.0.0.1:{f.server_address[1]}"

    tmp = tempfile.mkdtemp()
    pool = os.path.join(tmp, "pool.jsonl")
    docs = ["the quick brown fox runs and jumps over the lazy sleeping dog " * 30
            for _ in range(4)]
    with open(pool, "w", encoding="utf-8") as fh:
        for i, txt in enumerate(docs):
            select_rubric(txt)  # raises if rubric selection breaks on the sample
            fh.write(json.dumps({"sample_id": f"s{i:04d}", "language": "en",
                                 "length_band": "m", "source": "st/x", "url": None,
                                 "kind": "nl", "content": txt}) + "\n")
    out = os.path.join(tmp, "labels.jsonl")
    ledger = os.path.join(tmp, "ledger.jsonl")
    kept, rej, stats = drive(
        pool, out, [gu, fu], "stub-model", concurrency=4, timeout=10,
        limit=None, ledger=ledger)
    assert (kept, rej) == (4, 0), (kept, rej)
    assert stats[gu]["ok"] == 2 and stats[fu]["ok"] == 2, stats
    with open(out, encoding="utf-8") as fh:
        labels = [json.loads(l) for l in fh]
    assert len({l["sample_id"] for l in labels}) == 4
    for l in labels:
        assert set(l["scores"]) == {
            "content_quality", "factual_correctness",
            "complexity", "educational_or_code_value"}, l["scores"]
    with open(ledger, encoding="utf-8") as fh:
        n_ledger = sum(1 for _ in fh)
    assert n_ledger == 4, n_ledger
    kept2, _, _ = drive(pool, out, [gu, fu], "stub-model", concurrency=4,
                        timeout=10, limit=None, ledger=ledger)
    assert kept2 == 0, "resume did not skip done sample_ids"
    g.shutdown()
    f.shutdown()
    print("selftest ok: 4 labels across 2 endpoints (2 each), retry recovered "
          "HTTP500, 4 ledger rows, resume skipped all done")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--pool")
    ap.add_argument("--out")
    ap.add_argument("--urls")
    ap.add_argument("--model")
    ap.add_argument("--concurrency", type=int, default=256)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ledger", default=None)
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    missing = [n for n, v in (("--pool", a.pool), ("--out", a.out),
                              ("--urls", a.urls), ("--model", a.model)) if not v]
    if missing:
        ap.error("missing required: " + " ".join(missing))
    endpoints = [("http://" + u if not u.startswith("http") else u)
                 for u in a.urls.split(",") if u]
    drive(a.pool, a.out, endpoints, a.model, concurrency=a.concurrency,
          timeout=a.timeout, limit=a.limit, ledger=a.ledger)


if __name__ == "__main__":
    main()
