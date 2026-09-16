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
import contextlib
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

_CRASH_STATE = {"n": 0}  # selftest-only hard-kill hook state


def _post(url, model, timeout, text, rubric):
    base = url.rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    prompt, truncated = build_prompt(text, rubric)
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 200,
        # REQUIRED for Qwen3.x: without this the budget goes to CoT and no rubric JSON.
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(base + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"], truncated


def _label_one(item, endpoints, model, timeout):
    """item=(idx, row); returns (ok, payload, ep). One retry on a transient HTTP error."""
    idx, row = item
    ep = endpoints[idx % len(endpoints)]
    text = row["content"]
    rubric = select_rubric(text)
    last = None
    for _ in range(2):
        try:
            raw, truncated = _post(ep, model, timeout, text, rubric)
            scores = parse_scores(raw, rubric)
            return True, (row, scores, rubric["kind"], ep, truncated), ep
        except Exception as e:  # loud: persist the failure, never fabricate a label
            last = f"{type(e).__name__}: {e}"
    return False, (row.get("sample_id"), last, rubric["kind"], ep, False), ep


def _write_durable(out_fh, ledger_fh, label_row, ledger_row):
    """Persist one label as an atomic-across-crash unit. The score-ledger row is
    written and fsynced FIRST, then the label row is written and fsynced -- `out` is
    the commit marker. A SIGKILL at any instant therefore leaves every sample_id in
    `out` already durably present exactly once in the ledger (resume also treats ledger
    doc_ids as done, so the reverse window can not create a duplicate on restart).
    A write/fsync/validate failure raises and stops the run rather than continuing
    with the two files diverged (de, PR #400 block)."""
    if ledger_fh is not None:
        from score_ledger import validate_row
        validate_row(ledger_row)  # loud: never persist a schema-invalid ledger row
        ledger_fh.write(json.dumps(ledger_row, ensure_ascii=False, sort_keys=True) + "\n")
        ledger_fh.flush()
        os.fsync(ledger_fh.fileno())
    # test-only hard-kill hook (L3_DRIVE_CRASH_AFTER=N): die AFTER the ledger row is
    # durable but BEFORE the out row is fsynced -- exactly the window the ordering and
    # union-resume must survive. Never set in production.
    crash_n = os.environ.get("L3_DRIVE_CRASH_AFTER")
    if crash_n:
        _CRASH_STATE["n"] += 1
        if _CRASH_STATE["n"] == int(crash_n):
            os._exit(9)
    out_fh.write(json.dumps(label_row, ensure_ascii=False) + "\n")
    out_fh.flush()
    os.fsync(out_fh.fileno())


def _label_all(todo, out_fh, rej_fh, ledger_fh, endpoints, model, concurrency, timeout):
    lock = threading.Lock()
    stats = {ep: {"ok": 0, "rej": 0} for ep in endpoints}
    kept = rej = 0
    t0 = time.time()

    with cf.ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [ex.submit(_label_one, it, endpoints, model, timeout) for it in todo]
        for fu in cf.as_completed(futs):
            ok, payload, ep = fu.result()
            with lock:
                if ok:
                    row, scores, kind, _, truncated = payload
                    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                    label_id = hashlib.sha256(
                        (row["sample_id"] + model).encode()).hexdigest()[:16]
                    stratum = {"language": row.get("language"),
                               "length_band": row.get("length_band")}
                    label_row = {
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
                        "stratum": stratum,
                        "truncated": truncated,
                        "ts": ts,
                    }
                    ledger_row = None
                    if ledger_fh is not None:
                        ledger_row = {
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
                            "stratum": stratum,
                            "rubric_kind": kind,
                            "record_id": label_id,
                            "src_sha": None,
                            "truncated": truncated,
                        }
                    # a write/fsync failure raises and stops the run rather than
                    # continuing with out/ledger diverged
                    _write_durable(out_fh, ledger_fh, label_row, ledger_row)
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
    return kept, rej, stats, time.time() - t0


def drive(pool, out, endpoints, model, *, concurrency, timeout, limit, ledger):
    with open(pool, encoding="utf-8") as fh:
        rows = [json.loads(l) for l in fh]
    done = set()
    out_sids = set()
    if os.path.exists(out):
        with open(out, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    out_sids.add(json.loads(line)["sample_id"])
    done |= out_sids

    # The ledger is fsynced BEFORE its out mirror row, so a hard kill can leave a
    # doc_id durable in the ledger but absent from out. Reconcile on startup: rebuild
    # the missing out rows deterministically from the ledger joined to the pool (which
    # still carries source/url) -- no teacher re-call, no dropped/duplicated label.
    pool_by_sid = {r["sample_id"]: r for r in rows}
    ledger_doc_ids = set()
    recon = []
    if ledger and os.path.exists(ledger):
        with open(ledger, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                lr = json.loads(line)
                ledger_doc_ids.add(lr["doc_id"])
                if lr["doc_id"] not in out_sids:
                    recon.append(lr)
    if recon:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "a", encoding="utf-8") as rec_fh:
            for lr in recon:
                pr = pool_by_sid.get(lr["doc_id"], {})
                rec_fh.write(json.dumps({
                    "label_id": lr["record_id"],
                    "sample_id": lr["doc_id"],
                    "language": pr.get("language", lr.get("stratum", {}).get("language")),
                    "length_band": pr.get("length_band",
                                          lr.get("stratum", {}).get("length_band")),
                    "source": pr.get("source"),
                    "url": pr.get("url"),
                    "rubric_version": lr["scorer_version"],
                    "rubric_kind": lr["rubric_kind"],
                    "teacher_model": lr["model"],
                    "backend": lr["backend"],
                    "scores": lr["rubric_dims"],
                    "stratum": lr["stratum"],
                    "truncated": lr.get("truncated", False),
                    "ts": lr["ts"],
                }, ensure_ascii=False) + "\n")
            rec_fh.flush()
            os.fsync(rec_fh.fileno())
    done |= ledger_doc_ids
    todo = [(i, r) for i, r in enumerate(rows) if r["sample_id"] not in done]
    if limit is not None:
        todo = todo[:limit]

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    rej_path = out + ".rejected.jsonl"
    if ledger:
        os.makedirs(os.path.dirname(os.path.abspath(ledger)), exist_ok=True)
    with (
        open(out, "a", encoding="utf-8") as out_fh,
        open(rej_path, "a", encoding="utf-8") as rej_fh,
        (open(ledger, "a", encoding="utf-8") if ledger else contextlib.nullcontext()) as ledger_fh,
    ):
        result = _label_all(todo, out_fh, rej_fh, ledger_fh, endpoints, model,
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


def _crash_child(spec_path):
    """Selftest child: run drive and hard-die in the ledger->out window on row N."""
    with open(spec_path, encoding="utf-8") as fh:
        spec = json.load(fh)
    import http.server
    import socketserver

    class S(socketserver.TCPServer):
        allow_reuse_address = True

    class H(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("content-length", 0)))
            body = json.dumps({"choices": [{"message": {"content": json.dumps({
                "content_quality": 3, "factual_correctness": 3,
                "complexity": 3, "educational_or_code_value": 3})}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = S(("127.0.0.1", 0), H)
    import threading
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    if spec.get("crash_after"):
        os.environ["L3_DRIVE_CRASH_AFTER"] = str(spec["crash_after"])
    drive(spec["pool"], spec["out"], [url], "child-model", concurrency=1,
          timeout=10, limit=None, ledger=spec["ledger"])


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
    short = "the quick brown fox runs and jumps over the lazy sleeping dog " * 30
    long_doc = "ordinary prose words that keep repeating past the truncation limit " * 120
    assert len(short) <= 6000 < len(long_doc), (len(short), len(long_doc))
    docs = [short, short, short, long_doc]
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
    # truncated flag: only the >6000-char doc is a prefix label, in BOTH row kinds
    label_trunc = {l["sample_id"]: l["truncated"] for l in labels}
    assert label_trunc == {"s0000": False, "s0001": False,
                           "s0002": False, "s0003": True}, label_trunc
    with open(ledger, encoding="utf-8") as fh:
        led_rows = [json.loads(l) for l in fh]
    assert len(led_rows) == 4, len(led_rows)
    assert {r["doc_id"]: r["truncated"] for r in led_rows} == label_trunc
    kept2, _, _ = drive(pool, out, [gu, fu], "stub-model", concurrency=4,
                        timeout=10, limit=None, ledger=ledger)
    assert kept2 == 0, "resume did not skip done sample_ids"
    g.shutdown()
    f.shutdown()

    # ---- hard-kill durability: child dies in the ledger->out window, parent
    # reconciles + resumes; final out and ledger sets must be equal, no dupes/gaps.
    import subprocess
    d2 = tempfile.mkdtemp()
    pool2 = os.path.join(d2, "pool.jsonl")
    out2 = os.path.join(d2, "out.jsonl")
    led2 = os.path.join(d2, "ledger.jsonl")
    with open(pool2, "w", encoding="utf-8") as fh:
        for i in range(12):
            fh.write(json.dumps({"sample_id": f"k{i:03d}", "language": "en",
                                 "length_band": "m", "source": "st/x", "url": None,
                                 "kind": "nl",
                                 "content": "durable row content with many words " * 20})
                     + "\n")
    spec = os.path.join(d2, "spec.json")
    with open(spec, "w", encoding="utf-8") as fh:
        json.dump({"pool": pool2, "out": out2, "ledger": led2, "crash_after": 4}, fh)
    p = subprocess.run([sys.executable, os.path.abspath(__file__),
                        "--crash-child", spec], capture_output=True, timeout=60)
    assert p.returncode == 9, p.returncode  # hard exit in the dangerous window
    with open(out2, encoding="utf-8") as fh:
        n_out_killed = sum(1 for _ in fh)
    with open(led2, encoding="utf-8") as fh:
        n_led_killed = sum(1 for _ in fh)
    assert n_led_killed == 4 and n_out_killed == 3, (n_out_killed, n_led_killed)
    # restart in THIS process: reconcile the orphaned ledger row, finish the rest
    kept3, rej3, _ = _recover_and_finish(pool2, out2, led2)
    assert (kept3, rej3) == (8, 0), (kept3, rej3)  # 4 already scored, 8 remaining
    with open(out2, encoding="utf-8") as fh:
        out_ids = [json.loads(l)["sample_id"] for l in fh]
    with open(led2, encoding="utf-8") as fh:
        led_ids = [json.loads(l)["doc_id"] for l in fh]
    assert len(out_ids) == 12 and len(led_ids) == 12, (len(out_ids), len(led_ids))
    assert sorted(out_ids) == sorted(led_ids), "out/ledger diverged after recovery"
    assert len(set(out_ids)) == 12 and len(set(led_ids)) == 12, "duplicate label rows"
    print("selftest ok: 4 labels across 2 endpoints (2 each), retry recovered "
          "HTTP500, 4 ledger rows, resume skipped all done; SIGKILL at ledger->out "
          "row 4 recovered to out==ledger (12 each, no dup/gap)")
    return 0


def _recover_and_finish(pool, out, ledger):
    """selftest helper: relaunch a short-lived stub server, reconcile, finish pool."""
    import http.server
    import socketserver
    import threading

    class S(socketserver.TCPServer):
        allow_reuse_address = True

    class H(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("content-length", 0)))
            body = json.dumps({"choices": [{"message": {"content": json.dumps({
                "content_quality": 2, "factual_correctness": 2,
                "complexity": 2, "educational_or_code_value": 2})}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = S(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        return drive(pool, out, [f"http://127.0.0.1:{srv.server_address[1]}"],
                     "recover-model", concurrency=8, timeout=10, limit=None,
                     ledger=ledger)
    finally:
        srv.shutdown()


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
    ap.add_argument("--crash-child", default=None,
                    help="selftest-only: spec json; run drive and hard-exit in the "
                         "ledger->out window (L3_DRIVE_CRASH_AFTER).")
    a = ap.parse_args()
    if a.crash_child:
        _crash_child(a.crash_child)
        return 0
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
