#!/usr/bin/env python3
"""Frozen score-ledger schema: the single place every data-quality scorer writes.

# restartable: append-only JSONL; each row is validated then written as one line, so an
# interrupt costs at most the unflushed tail and re-running append re-adds rows without
# touching prior ones. It never scans or rewrites a corpus shard.

Scorer-agnostic and append-only. KenLM, FineWeb-Edu, and the L3 teacher rubric all append
rows here; one document can have one row per (scorer_name, scorer_version[, rubric_kind]),
and rows are never rewritten -- a re-score is a NEW row with a new scorer_version. The
per-domain quota/threshold selector is a later module that only READS this ledger.

Schema (one JSON object per line, all keys present every line):

    doc_id         str  non-empty. Stable per-document CONTENT id.
                       Census (full-corpus) scores: content_doc_id(text) =
                       sha256(content)[:16]. Sampled scores: the sampler's sample_id.
    domain         str  non-empty corpus domain (mix domain name, e.g. zh_web, py).
    lang           str  non-empty BCP-47-ish language code (en, zh, ...).
    scorer_name    str  non-empty (kenlm | fineweb-edu | l3-rubric | ...).
    scorer_version str  non-empty version pin; a retrained/drifting scorer bumps this.
    ts             str  score time, ISO-8601 UTC, MUST end in 'Z'.
    score          float | None. One continuous scalar score.
    rubric_dims    dict[str, int] | None. Multi-dimensional rubric; every value is an int
                       in 1..5. Exactly ONE of score / rubric_dims is present.
    cut            float | None. The keep threshold THIS score row was produced against,
                       when the scorer carries one (fineweb-edu cut>=3); else None.
    model          str | None. Model/teacher identifier (HF id, teacher model, lm file).
    backend        str | None. How it was scored (openai | local-cpu | stub | ...).
    stratum        dict | None. Sampling stratum the row was drawn from. None means a
                       CENSUS row (every document was scored), not "unknown": a row whose
                       provenance is unknown must not be written.
    rubric_kind    str | None. Rubric variant for a multi-dim row (66 l3-rubric kind).
    record_id      str | None. External id used to JOIN the row back to an upstream record
                       that is not addressable by the content hash: a sampling run's
                       label_id (66), or an existing labeled set's own source id (the cci3
                       audit ids are a source-repo hash, not sha256(content), so a locked-
                       set rescoring stores that id here to join back to the hand labels).
                       doc_id stays the content identity; record_id is the provenance-set
                       join handle. Null for a census row with no external record.
    src_sha        str | None. Content fingerprint of the CORPUS SOURCE BUILD the document
                       belongs to (the corpus_fingerprint/filters_fp convention), a 64-hex
                       sha256. Optional: census rows over a fingerprinted build fill it.

Content identity / staleness (the reason doc_id is a content hash): if the source is
rebuilt or re-cleaned the document content changes, content_doc_id changes, so the old
row no longer joins and is naturally stale -- it is never silently read as the score of
new content. assert_doc_matches_content() re-hashes to enforce this at write time when the
text is in hand. src_sha names the build; it is a different granularity (whole corpus)
from doc_id (one document), so the ledger cannot compare the two for equality -- it only
type-checks src_sha, and the caller ties a row to content via doc_id.

    python3 datagen/score_ledger.py append ledger.jsonl rows.json
    python3 datagen/score_ledger.py --selftest
"""

import argparse
import hashlib
import json
import math
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
DOC_ID_LEN = 16
RUBRIC_MIN, RUBRIC_MAX = 1, 5

REQUIRED_STR = ("doc_id", "domain", "lang", "scorer_name", "scorer_version", "ts")
ALL_FIELDS = (
    *REQUIRED_STR,
    "score",
    "rubric_dims",
    "cut",
    "model",
    "backend",
    "stratum",
    "rubric_kind",
    "record_id",
    "src_sha",
)


class LedgerSchemaError(ValueError):
    """A row violates the frozen schema; writers must refuse, never silently persist it."""


def content_doc_id(text: str) -> str:
    """The census doc id: sha256 of the document CONTENT (utf-8), first 16 hex chars."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:DOC_ID_LEN]


def assert_doc_matches_content(doc_id: str, text: str):
    """Re-hash content and refuse a doc_id that does not match. The load-bearing guard that
    keeps a score tied to exactly the bytes it was computed on."""
    actual = content_doc_id(text)
    if doc_id != actual:
        raise LedgerSchemaError(
            f"doc_id {doc_id!r} does not match content hash {actual!r}; the document "
            "changed, so a score for it must be written under the new doc_id"
        )


def _is_int_1_5(v):
    # bool is an int subclass; a rubric grade is never a bool
    return isinstance(v, int) and not isinstance(v, bool) and RUBRIC_MIN <= v <= RUBRIC_MAX


def _utc_z(ts):
    if not isinstance(ts, str) or not ts.endswith("Z"):
        return False
    try:
        datetime.fromisoformat(ts[:-1] + "+00:00")
    except ValueError:
        return False
    return True


def _finite_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def validate_row(row):
    """Validate one ledger row. Raises LedgerSchemaError naming the first violation."""
    if not isinstance(row, dict):
        raise LedgerSchemaError(f"row must be an object, got {type(row).__name__}")

    for f in REQUIRED_STR:
        v = row.get(f)
        if not isinstance(v, str) or not v:
            raise LedgerSchemaError(f"{f} must be a non-empty string, got {row.get(f)!r}")
    if not _utc_z(row["ts"]):
        raise LedgerSchemaError(f"ts must be ISO-8601 UTC ending in 'Z', got {row['ts']!r}")

    score = row.get("score")
    dims = row.get("rubric_dims")
    has_score = score is not None
    has_dims = dims is not None
    if has_score == has_dims:
        raise LedgerSchemaError(
            f"exactly one of score / rubric_dims must be present (score={score!r}, rubric_dims={dims!r})"
        )
    if has_score and not _finite_number(score):
        raise LedgerSchemaError(f"score must be a finite number, got {score!r}")
    if has_dims:
        if not isinstance(dims, dict) or not dims:
            raise LedgerSchemaError("rubric_dims must be a non-empty dict")
        for k, v in dims.items():
            if not isinstance(k, str) or not k:
                raise LedgerSchemaError(f"rubric dim names must be non-empty strings: {k!r}")
            if not _is_int_1_5(v):
                raise LedgerSchemaError(
                    f"rubric_dims[{k!r}] must be an int in {RUBRIC_MIN}..{RUBRIC_MAX}, got {v!r}"
                )

    cut = row.get("cut")
    if cut is not None and not _finite_number(cut):
        raise LedgerSchemaError(f"cut must be a finite number or null, got {cut!r}")

    stratum = row.get("stratum")
    if stratum is not None and not isinstance(stratum, dict):
        raise LedgerSchemaError("stratum must be a dict or null (null=census, not unknown)")

    src_sha = row.get("src_sha")
    if src_sha is not None and not (isinstance(src_sha, str) and _HEX64.match(src_sha)):
        raise LedgerSchemaError(f"src_sha must be a 64-hex sha256 or null, got {src_sha!r}")

    for f in ("model", "backend", "rubric_kind", "record_id"):
        v = row.get(f)
        if v is not None and not isinstance(v, str):
            raise LedgerSchemaError(f"{f} must be a string or null, got {v!r}")


@dataclass
class ScoreRow:
    doc_id: str
    domain: str
    lang: str
    scorer_name: str
    scorer_version: str
    ts: str
    score: object = None
    rubric_dims: object = None
    cut: object = None
    model: object = None
    backend: object = None
    stratum: object = None
    rubric_kind: object = None
    record_id: object = None
    src_sha: object = None

    def to_dict(self):
        d = asdict(self)
        validate_row(d)
        return d


def append_rows(path, rows):
    """Validate and append rows (dicts or ScoreRow) to an append-only JSONL ledger."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    n = 0
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            d = row.to_dict() if isinstance(row, ScoreRow) else dict(row)
            validate_row(d)
            f.write(json.dumps(d, ensure_ascii=False, sort_keys=True) + "\n")
            n += 1
    return n


def load_rows(path):
    """Read and validate every ledger row. A truncated final line raises rather than
    silently dropping a score."""
    out = []
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise LedgerSchemaError(f"{path}:{ln}: malformed ledger line: {e}") from e
            validate_row(row)
            out.append(row)
    return out


def _selftest():
    import tempfile

    valid_scalar = {
        "doc_id": "9f3ac1d4e0b72a6f",
        "domain": "zh_web",
        "lang": "zh",
        "scorer_name": "fineweb-edu",
        "scorer_version": "2024-06",
        "ts": "2026-09-16T08:00:00Z",
        "score": 0.86,
        "rubric_dims": None,
        "cut": 3.0,
        "model": "HuggingFaceFW/fineweb-edu-classifier",
        "backend": None,
        "stratum": None,
        "rubric_kind": None,
        "record_id": None,
        "src_sha": "a" * 64,
    }
    valid_rubric = {
        "doc_id": "s1",
        "domain": "py",
        "lang": "en",
        "scorer_name": "l3-rubric",
        "scorer_version": "r1",
        "ts": "2026-09-16T08:00:00Z",
        "score": None,
        "rubric_dims": {"content_quality": 4, "complexity": 2},
        "cut": None,
        "model": "teacher",
        "backend": "openai",
        "stratum": {"language": "en", "length_band": "m"},
        "rubric_kind": "py",
        "record_id": "lab1",
        "src_sha": None,
    }
    validate_row(valid_scalar)
    validate_row(valid_rubric)
    ScoreRow(**{k: v for k, v in valid_scalar.items()}).to_dict()

    # each invalid category must FAIL on its own, by name
    def m_empty_domain(r):
        r["domain"] = ""

    def m_bad_ts(r):
        r["ts"] = "2026-09-16T08:00:00"  # no Z / not UTC

    def m_both_present(r):
        r["rubric_dims"] = {"a": 1}

    def m_neither(r):
        r["score"] = None

    def m_nan(r):
        r["score"] = float("nan")

    def m_bool(r):
        r["score"] = True

    def m_dims_range(r):
        r["score"] = None
        r["rubric_dims"] = {"a": 6}

    def m_dims_bool(r):
        r["score"] = None
        r["rubric_dims"] = {"a": True}

    def m_bad_cut(r):
        r["cut"] = float("inf")

    def m_bad_stratum(r):
        r["stratum"] = "unknown"

    def m_bad_srcsha(r):
        r["src_sha"] = "deadbeef"

    def m_bad_model(r):
        r["model"] = 7

    def bad(mut, label):
        r = json.loads(json.dumps(valid_scalar))
        mut(r)
        try:
            validate_row(r)
        except LedgerSchemaError:
            return
        raise AssertionError(f"bad row accepted: {label}")

    for mut, label in (
        (m_empty_domain, "empty domain"),
        (m_bad_ts, "ts not UTC Z"),
        (m_both_present, "score+dims both"),
        (m_neither, "score+dims neither"),
        (m_nan, "nan score"),
        (m_bool, "bool score"),
        (m_dims_range, "rubric 1..5"),
        (m_dims_bool, "rubric bool"),
        (m_bad_cut, "inf cut"),
        (m_bad_stratum, "stratum type"),
        (m_bad_srcsha, "src_sha hex"),
        (m_bad_model, "model type"),
    ):
        bad(mut, label)

    # append/load round trip + content-hash identity and staleness guard
    with tempfile.TemporaryDirectory() as td:
        led = os.path.join(td, "ledger.jsonl")
        assert append_rows(led, [valid_scalar, valid_rubric]) == 2
        assert len(load_rows(led)) == 2
        cid = content_doc_id("print('hi')\n")
        assert_doc_matches_content(cid, "print('hi')\n")
        try:
            assert_doc_matches_content(cid, "print('bye')\n")
        except LedgerSchemaError:
            pass
        else:
            raise AssertionError("changed content must not match the old doc_id")

    print(
        "score_ledger selftest OK: 2 valid row shapes accepted, 12 invalid categories "
        "rejected, append/load round-trip, content-hash staleness guard"
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("append")
    p.add_argument("ledger")
    p.add_argument("rows_json")
    a = ap.parse_args(argv)
    if a.selftest:
        _selftest()
        return
    if a.cmd == "append":
        with open(a.rows_json, encoding="utf-8") as f:
            rows = json.load(f)
        print(f"appended {append_rows(a.ledger, rows)} rows -> {a.ledger}")
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
