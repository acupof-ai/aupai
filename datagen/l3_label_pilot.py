"""Orchestrate L3 multi-dimensional teacher labels over a stratified pilot.

# restartable: resumable by sample_id — completed doc ids are read from --out on start
# and skipped, labels/rejections are appended per row, so an interrupt re-sends only
# the unfinished tail; the ledger double-write is a post-pass append over this run's rows.

CPU orchestration only. The teacher call is pluggable:
- --backend stub   deterministic, no network (used to run the pipeline + parser end
                   to end on a few hundred rows now; NOT a quality label, scores are a
                   fixed function of content length so parsing/versioning is exercised).
- --backend openai  an OpenAI-compatible /v1/chat/completions endpoint on the pod
                   (the existing 27B servers; --url/--model), the real teacher.

Output jsonl, one label per input row, identity-versioned for later distillation:
{label_id, sample_id, language, length_band, source, url, rubric_version, rubric_kind,
 teacher_model, backend, scores:{4 dims}, stratum:{...}, ts}. Rows the teacher/parser
reject are written to <out>.rejected.jsonl with the raw reply + reason, never dropped
silently, and counted in the manifest. Resume: already-labelled sample_ids are skipped.

Small pilot first (a few hundred); do NOT point this at millions.
"""

import argparse
import hashlib
import json
import os
import time
import urllib.request

from l3_rubric import RUBRIC_VERSION, TRUNC_CHARS, build_prompt, parse_scores, select_rubric


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


def make_openai_teacher(url, model, timeout):
    # accept either ".../v1" or a bare root; append the chat path once.
    base = url.rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    endpoint = base + "/chat/completions"

    def ask(text, rubric):
        prompt, _ = build_prompt(text, rubric)
        body = json.dumps(
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 200,
                # REQUIRED for Qwen3.5: without enable_thinking=false the model spends
                # the whole budget on a chain-of-thought trace and returns no parseable
                # rubric JSON.
                "chat_template_kwargs": {"enable_thinking": False},
            }
        ).encode()
        req = urllib.request.Request(endpoint, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read())
        return d["choices"][0]["message"]["content"]

    return ask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", required=True, help="stratified sample jsonl")
    ap.add_argument("--out", required=True, help="labels jsonl")
    ap.add_argument("--backend", choices=["stub", "openai"], default="stub")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="teacher")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--limit", type=int, default=None, help="cap rows this run (pilot)")
    ap.add_argument(
        "--ledger",
        default=None,
        help="optional frozen score-ledger jsonl to ALSO append rows to "
        "(datagen/score_ledger schema, validated on write)",
    )
    ap.add_argument(
        "--domain",
        default=None,
        help="full corpus domain name for ledger rows, e.g. en_c4_stage2_dc; required when --ledger is set",
    )
    args = ap.parse_args()

    ledger_rows = []
    if args.ledger and not args.domain:
        raise SystemExit("REFUSE: --ledger requires --domain (full corpus domain name)")

    teacher = (
        stub_teacher if args.backend == "stub" else make_openai_teacher(args.url, args.model, args.timeout)
    )
    teacher_id = "stub-length-function" if args.backend == "stub" else args.model

    done = set()
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as fh:
            for line in fh:
                done.add(json.loads(line)["sample_id"])

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    rej_path = args.out + ".rejected.jsonl"
    kept = rej = 0
    with (
        open(args.pilot, encoding="utf-8") as inp,
        open(args.out, "a", encoding="utf-8") as out,
        open(rej_path, "a", encoding="utf-8") as rej_fh,
    ):
        for line in inp:
            row = json.loads(line)
            sid = row["sample_id"]
            if sid in done:
                continue
            if args.limit is not None and (kept + rej) >= args.limit:
                break
            text = row["content"]
            rubric = select_rubric(text)
            # build_prompt cuts and flags on the same predicate; the openai teacher gets
            # the flagged prompt, and this bool is threaded onto both emitted rows.
            truncated = len(text) > TRUNC_CHARS
            try:
                raw = teacher(text, rubric)
                scores = parse_scores(raw, rubric)
            except Exception as e:  # loud: persist, count, never fabricate
                rej_fh.write(
                    json.dumps(
                        {"sample_id": sid, "reason": str(e), "rubric_kind": rubric["kind"], "ts": _ts()},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                rej += 1
                continue
            ts = _ts()
            label_id = hashlib.sha256((sid + teacher_id).encode()).hexdigest()[:16]
            out.write(
                json.dumps(
                    {
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
                        "stratum": {"language": row.get("language"), "length_band": row.get("length_band")},
                        "truncated": truncated,
                        "ts": ts,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            if args.ledger:
                ledger_rows.append(
                    {
                        "doc_id": sid,
                        "domain": args.domain,
                        "lang": str(row.get("language") or "_"),
                        "scorer_name": "l3-rubric",
                        "scorer_version": RUBRIC_VERSION,
                        "ts": ts,
                        "score": None,
                        "rubric_dims": scores,
                        "cut": None,
                        "model": teacher_id,
                        "backend": args.backend,
                        "stratum": {"language": row.get("language"), "length_band": row.get("length_band")},
                        "rubric_kind": rubric["kind"],
                        "record_id": label_id,
                        "src_sha": None,
                        "truncated": truncated,
                    }
                )
            kept += 1

    ledger_n = 0
    if args.ledger and ledger_rows:
        from datagen.score_ledger import append_rows  # clean dependency on #389/main

        ledger_n = append_rows(args.ledger, ledger_rows)

    manifest = {
        "pilot": args.pilot,
        "out": args.out,
        "backend": args.backend,
        "teacher_model": teacher_id,
        "rubric_version": RUBRIC_VERSION,
        "kept": kept,
        "rejected": rej,
        "rejected_path": rej_path,
        "ledger": args.ledger,
        "ledger_domain": args.domain,
        "ledger_appended": ledger_n,
        "ts": _ts(),
    }
    with open(args.out + ".label_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
