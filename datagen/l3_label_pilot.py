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


def read_jsonl_lenient(path, id_key):
    """Read a JSONL of records, returning (ids, torn_tail, total_valid).

    Resume must SURVIVE a crash that tore the final line: a partial last record is not
    valid JSON, is not a label, and would make strict json.loads abort forever. So:
    - every non-last record must be valid JSON carrying id_key; a bad record in the
      MIDDLE is corruption (not a torn append) and raises loudly;
    - only a single unparseable LAST non-empty line is tolerated and reported as
      torn_tail=True so the caller can truncate it and re-label that sample.
    """
    ids = []
    torn_tail = False
    if not os.path.exists(path):
        return ids, False
    with open(path, encoding="utf-8") as fh:
        raw = [ln for ln in (line.strip() for line in fh) if ln]
    for i, rec in enumerate(raw):
        is_last = i == len(raw) - 1
        try:
            obj = json.loads(rec)
        except json.JSONDecodeError:
            if is_last:
                torn_tail = True
                continue
            raise SystemExit(
                f"REFUSE resume: {path} has invalid JSON on line {i + 1} of {len(raw)} "
                "(a non-final record is corrupt, not a torn append). Inspect/recover "
                "manually; only a single partial LAST line is auto-truncated.") from None
        ids.append(obj[id_key])
    return ids, torn_tail


def truncate_torn_tail(path):
    """Drop a single partial (non-JSON) trailing line so an append-mode resume rewrites
    that record whole. Rewritten atomically via tmp+replace; returns True if truncated."""
    if not os.path.exists(path):
        return False
    with open(path, encoding="utf-8") as fh:
        lines = fh.readlines()
    kept = [ln for ln in lines if ln.strip()]
    if not kept:
        return False
    try:
        json.loads(kept[-1])
        return False
    except json.JSONDecodeError:
        kept = kept[:-1]
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.writelines(kept)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        return True


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


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_resume_identity(out, pilot_path, pilot_sha, config):
    """Refuse a resume unless the existing --out was produced against the SAME pilot
    bytes and SAME backend/model/rubric version. Sequence-mode sample_ids are
    positional, so a changed pilot silently rebinds every id to different content; a
    changed teacher/backend mixes labels. Returns the count already labelled.

    Raises SystemExit on any mismatch. No prior labels (empty out) is a fresh run."""
    if not os.path.exists(out):
        return 0
    ids, _torn = read_jsonl_lenient(out, "sample_id")
    done = set(ids)
    if not done:
        return 0
    manifest_path = out + ".label_manifest.json"
    if not os.path.exists(manifest_path):
        raise SystemExit(
            f"REFUSE resume: {out} has {len(done)} labels but {manifest_path} is "
            "missing, so pilot/config identity cannot be verified. Use a new --out.")
    with open(manifest_path, encoding="utf-8") as fh:
        prior = json.load(fh)
    if prior.get("pilot_sha256") != pilot_sha:
        raise SystemExit(
            f"REFUSE resume: pilot file changed.\n"
            f"  manifest pilot_sha256: {prior.get('pilot_sha256')}\n"
            f"  current  pilot_sha256: {pilot_sha}\n"
            "Sequence-mode sample_ids are positional; the same id now points at "
            "different content. Re-draw to a new --out instead of resuming.")
    for key in ("backend", "teacher_model", "rubric_version"):
        if prior.get(key) != config[key]:
            raise SystemExit(
                f"REFUSE resume: {key} differs from the prior run "
                f"(manifest={prior.get(key)!r}, current={config[key]!r}). "
                "Mixing teachers/backends/rubric versions under one --out is rejected; "
                "label to a new --out.")
    return len(done)


def _selftest():
    import tempfile

    def expect_refuse(label, out, pilot, sha, cfg):
        try:
            check_resume_identity(out, pilot, sha, cfg)
        except SystemExit:
            return
        raise AssertionError(f"{label}: mismatched resume must be refused")

    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "labels.jsonl")
        pilot = os.path.join(td, "pilot.jsonl")
        with open(pilot, "w", encoding="utf-8") as fh:
            fh.write('{"sample_id": "nl-s-0000000", "content": "hello world"}\n')
        sha = file_sha256(pilot)
        cfg = {"backend": "openai", "teacher_model": "m", "rubric_version": RUBRIC_VERSION}
        man = out + ".label_manifest.json"

        # fresh: no out file -> 0
        assert check_resume_identity(out, pilot, sha, cfg) == 0

        # simulate a completed prior run: 2 labels + matching manifest
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"sample_id": "a"}) + "\n")
            fh.write(json.dumps({"sample_id": "b"}) + "\n")
        with open(man, "w", encoding="utf-8") as fh:
            json.dump({"pilot_sha256": sha, **cfg}, fh)
        assert check_resume_identity(out, pilot, sha, cfg) == 2  # identical -> resumes

        # broken world 1: pilot bytes changed (positional ids rebound) -> refuse
        with open(pilot, "w", encoding="utf-8") as fh:
            fh.write('{"sample_id": "nl-s-0000000", "content": "DIFFERENT content"}\n')
        new_sha = file_sha256(pilot)
        assert new_sha != sha
        expect_refuse("changed pilot", out, pilot, new_sha, cfg)

        # broken world 2: backend / teacher / rubric mismatch -> refuse each
        with open(man, "w", encoding="utf-8") as fh:
            json.dump({"pilot_sha256": new_sha, **cfg}, fh)
        for key, bad in (("backend", "stub"), ("teacher_model", "other"),
                         ("rubric_version", "future-v9")):
            badcfg = dict(cfg)
            badcfg[key] = bad
            expect_refuse(f"changed {key}", out, pilot, new_sha, badcfg)

        # broken world 3: labels present but manifest gone -> refuse (cannot verify)
        os.remove(man)
        expect_refuse("missing manifest", out, pilot, new_sha, cfg)

    # --- torn-tail self-heal and rejected dedup (audit #18/#19) in a fresh dir ---
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "labels.jsonl")
        rej = out + ".rejected.jsonl"

        # a torn LAST label line: valid preceding rows load, tail flagged + truncated,
        # and the torn sample is no longer in the done set so it gets retried.
        with open(out, "w", encoding="utf-8") as fh:
            fh.write('{"sample_id": "good1"}\n')
            fh.write('{"sample_id": "good2"}\n')
            fh.write('{"sample_id": "tor')  # partial, no newline
        ids, torn = read_jsonl_lenient(out, "sample_id")
        assert ids == ["good1", "good2"] and torn is True
        assert truncate_torn_tail(out) is True
        with open(out, encoding="utf-8") as rf:
            repaired = rf.read().splitlines()
        assert repaired == ['{"sample_id": "good1"}', '{"sample_id": "good2"}']
        assert truncate_torn_tail(out) is False  # idempotent: nothing torn now

        # a bad MIDDLE record must stay loud (never silently dropped)
        with open(out, "w", encoding="utf-8") as fh:
            fh.write('{"sample_id": "good1"}\n')
            fh.write('NOT JSON IN THE MIDDLE\n')
            fh.write('{"sample_id": "good3"}\n')
        try:
            read_jsonl_lenient(out, "sample_id")
        except SystemExit:
            pass
        else:
            raise AssertionError("corrupt middle record must refuse")

        # rejected dedup: two prior rejects load into the seen set; a torn reject tail
        # is truncated and not counted.
        with open(rej, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"sample_id": "r1"}) + "\n")
            fh.write(json.dumps({"sample_id": "r2"}) + "\n")
            fh.write('{"sample_id": "r3-tor')  # partial reject
        rids, rtorn = read_jsonl_lenient(rej, "sample_id")
        assert rids == ["r1", "r2"] and rtorn is True
        assert truncate_torn_tail(rej) is True
        assert set(read_jsonl_lenient(rej, "sample_id")[0]) == {"r1", "r2"}
    print("selftest ok: fresh=0; identical resume allowed; changed pilot, changed "
          "backend/model/rubric, and missing manifest all refused; torn LAST line in "
          "labels/rejects self-heals (middle corruption loud); rejected ids dedup on "
          "resume so a re-run never doubles a reject")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--pilot", help="stratified sample jsonl")
    ap.add_argument("--out", help="labels jsonl")
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
    if args.selftest:
        return _selftest()
    if not args.pilot or not args.out:
        ap.error("--pilot and --out are required")

    ledger_rows = []
    if args.ledger and not args.domain:
        raise SystemExit("REFUSE: --ledger requires --domain (full corpus domain name)")

    teacher = (
        stub_teacher if args.backend == "stub" else make_openai_teacher(args.url, args.model, args.timeout)
    )
    teacher_id = "stub-length-function" if args.backend == "stub" else args.model

    cur_config = {"backend": args.backend, "teacher_model": teacher_id,
                  "rubric_version": RUBRIC_VERSION}
    cur_pilot_sha = file_sha256(args.pilot)
    # #411 guard: refuse resume across changed pilot bytes or teacher/config. Run before
    # the lenient/torn-tail self-heal so identity is verified on the prior manifest first.
    check_resume_identity(args.out, args.pilot, cur_pilot_sha, cur_config)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    rej_path = args.out + ".rejected.jsonl"

    # Self-heal torn suffixes from a crash mid-append before resuming: a partial LAST
    # line in labels or rejects is truncated (atomic rewrite) so the sample is retried;
    # a bad MIDDLE record stays a loud refusal inside read_jsonl_lenient.
    label_ids, label_torn = read_jsonl_lenient(args.out, "sample_id")
    done = set(label_ids)
    truncated_labels = truncate_torn_tail(args.out) if label_torn else False
    rej_ids, rej_torn = read_jsonl_lenient(rej_path, "sample_id")
    rejected_seen = set(rej_ids)
    truncated_rejects = truncate_torn_tail(rej_path) if rej_torn else False

    kept = rej = 0
    with (
        open(args.pilot, encoding="utf-8") as inp,
        open(args.out, "a", encoding="utf-8") as out,
        open(rej_path, "a", encoding="utf-8") as rej_fh,
    ):
        for line in inp:
            row = json.loads(line)
            sid = row["sample_id"]
            if sid in done or sid in rejected_seen:
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
                if sid in rejected_seen:
                    continue  # defensive: already counted on this resume's input set
                rej_fh.write(
                    json.dumps(
                        {"sample_id": sid, "reason": str(e), "rubric_kind": rubric["kind"], "ts": _ts()},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                rejected_seen.add(sid)
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

    # manifest counts describe what is ON DISK after this run (prior + new), so the
    # manifest never drifts from the labels/rejected file line counts after a resume.
    total_kept = len(done)
    total_rej = len(rejected_seen)
    manifest = {
        "pilot": args.pilot,
        "pilot_sha256": cur_pilot_sha,
        "out": args.out,
        "backend": args.backend,
        "teacher_model": teacher_id,
        "rubric_version": RUBRIC_VERSION,
        "kept": total_kept,
        "newly_kept_this_run": kept,
        "rejected": total_rej,
        "newly_rejected_this_run": rej,
        "rejected_path": rej_path,
        "repaired_torn_label_tail": truncated_labels,
        "repaired_torn_reject_tail": truncated_rejects,
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
