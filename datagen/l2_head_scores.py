#!/usr/bin/env python3
"""Canonical rows for the L2 quality head's CONTINUOUS census predictions.

# restartable: append-only per-(domain,shard) JSONL; each row is validated then written as
# one line, so an interrupt costs only the unflushed tail and re-running appends without
# touching prior rows. It never scans or rewrites a corpus shard.

These are NOT teacher labels. The frozen score ledger (datagen/score_ledger.py) is for the
66 teacher's discrete int 1..5 grades (rubric_dims locked to int) and single-scalar scorer
rows; the L2 head instead emits four CONTINUOUS float predictions on the same rubric axes,
and a value can legitimately sit outside 1..5 (it is the head's raw native scale, not a
grade). Folding them into score_ledger would force a clamp/round and corrupt the signal, so
they get this sibling schema (fb ruling 2026-09-16).

Row (one JSON object per line, every key present):

    doc_id        str  sha256(content)[:16] of the WHOLE scored document (the encoder scores
                  up to 8192 tokens of the document, not a 1024 pool chunk), via
                  score_ledger.content_doc_id.
    domain, lang  str  mix domain + language of the document
    scorer_name   str  "l2-head"
    scorer_version str head/checkpoint version pin; one selection pins exactly one
    ts            str  ISO-8601 UTC ending in "Z"
    rubric_kind   str  "code" | "natural_language"; selects the dim key set
    model         str  the encoder+head identifier actually run
    source_shard  str | None  source corpus shard for provenance
    src_sha       str | None  64-hex corpus-build fingerprint (same meaning as in ledger)
    dim_scale     str  "rubric_1_5" -- the numeric scale the dims live on; a future scale
                  must bump this, it is not silently reinterpreted
    dims          {dim: finite float} EXACTLY the four keys of
                  l3_rubric.RUBRIC_DIMS[rubric_kind]; floats are NOT clamped or rounded.

Selection reuses the frozen quota machinery with no duplicated ordering math: build_groups
adapts these rows to score_quota's {(domain, lang): [(doc_id, value)]} shape (one dim, or
the mean of the four), then score_quota.select_thresholds / export_review_sample apply the
same per-domain conditional quota, deterministic tie-break, and empty/wiped/version guards.

    python3 datagen/l2_head_scores.py append preds.jsonl rows.json
    python3 datagen/l2_head_scores.py --selftest
"""

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datagen import score_quota  # noqa: E402
from datagen.l3_rubric import CODE_RUBRIC, NL_RUBRIC  # noqa: E402

DIM_SCALE = "rubric_1_5"
SCORER = "l2-head"

_RUBRIC_DIMS = {
    CODE_RUBRIC["kind"]: tuple(CODE_RUBRIC["dimensions"]),
    NL_RUBRIC["kind"]: tuple(NL_RUBRIC["dimensions"]),
}
REQUIRED_STR = (
    "doc_id",
    "domain",
    "lang",
    "scorer_name",
    "scorer_version",
    "ts",
    "rubric_kind",
    "model",
    "dim_scale",
)
_HEX64 = __import__("re").compile(r"^[0-9a-f]{64}$")


class HeadScoreError(ValueError):
    """A head-score row violates the continuous-prediction schema."""


def _utc_z(ts):
    if not isinstance(ts, str) or not ts.endswith("Z"):
        return False
    from datetime import datetime

    try:
        datetime.fromisoformat(ts[:-1] + "+00:00")
    except ValueError:
        return False
    return True


def _finite(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def validate_row(row):
    if not isinstance(row, dict):
        raise HeadScoreError(f"row must be an object, got {type(row).__name__}")
    for f in REQUIRED_STR:
        v = row.get(f)
        if not isinstance(v, str) or not v:
            raise HeadScoreError(f"{f} must be a non-empty string, got {row.get(f)!r}")
    if not _utc_z(row["ts"]):
        raise HeadScoreError(f"ts must be ISO-8601 UTC ending in Z, got {row['ts']!r}")
    if row["dim_scale"] != DIM_SCALE:
        raise HeadScoreError(
            f"dim_scale must be {DIM_SCALE!r} (native continuous rubric scale); got "
            f"{row['dim_scale']!r} -- bump the schema for a new scale, do not reinterpret"
        )
    kind = row["rubric_kind"]
    expected = _RUBRIC_DIMS.get(kind)
    if expected is None:
        raise HeadScoreError(f"rubric_kind {kind!r} not in l3_rubric {sorted(_RUBRIC_DIMS)}")
    dims = row.get("dims")
    if not isinstance(dims, dict):
        raise HeadScoreError("dims must be an object of four finite floats")
    if set(dims) != set(expected):
        raise HeadScoreError(f"dims keys for {kind} must be exactly {list(expected)}, got {sorted(dims)}")
    for k in expected:
        if not _finite(dims[k]):
            raise HeadScoreError(f"dims[{k!r}] must be a finite float, got {dims[k]!r}")
    for f in ("source_shard", "src_sha"):
        v = row.get(f)
        if v is not None and not isinstance(v, str):
            raise HeadScoreError(f"{f} must be a string or null, got {v!r}")
    if row["src_sha"] is not None and not _HEX64.match(row["src_sha"]):
        raise HeadScoreError(f"src_sha must be a 64-hex sha256 or null, got {row['src_sha']!r}")


def append_rows(path, rows):
    """Validate and append continuous head-score rows to an append-only JSONL file,
    all-or-nothing for the cheap deterministic failures: EVERY row is validated and
    serialized BEFORE the file is opened, so one bad/unserializable row raises with zero
    bytes from this batch touching disk -- the file's prior prefix is untouched and re-running
    never double-writes the good rows that preceded the bad one. Once the whole batch is
    known-good it is appended in one open and flushed+fsynced before close so a completed
    call's bytes are durable. Crash-safe replacement of the shard itself (tmp+rename+dir
    fsync) is the caller's job; this guarantees no partial VALIDATION prefix."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    # materialize + validate + serialize up front: a generator or list is fully consumed here,
    # and any HeadScoreError/TypeError lands before the append handle exists.
    lines = []
    for row in rows:
        validate_row(row)
        lines.append(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    if not lines:
        return 0
    with open(path, "a", encoding="utf-8") as f:
        f.writelines(lines)
        f.flush()
        os.fsync(f.fileno())
    return len(lines)


def load_rows(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise HeadScoreError(f"{path}:{ln}: malformed row: {e}") from e
            validate_row(row)
            out.append(row)
    return out


def _row_value(row, rubric_dim):
    dims = row["dims"]
    if rubric_dim is not None:
        return float(dims[rubric_dim])
    return sum(dims.values()) / len(dims)


def build_groups(rows, scorer_version=None, rubric_dim=None, by_lang=True):
    """Adapt validated head-score rows to score_quota's group shape, ordering on one dim or
    the four-dim MEAN. Pins one scorer_version (multiple present -> score_quota.VersionConflict
    raises). Returns {(domain, lang): [(doc_id, value), ...]} plus the resolved version.

    The float value is passed through UNQUANTIZED; this is what keeps the quota ranking on
    the head's continuous signal rather than a rounded grade."""
    versions = sorted({r["scorer_version"] for r in rows})
    if not rows:
        raise score_quota.EmptyDomain("no l2-head score rows")
    if scorer_version is not None:
        if scorer_version not in versions:
            raise score_quota.VersionConflict(f"requested {scorer_version!r}, have {versions}")
        sel = [r for r in rows if r["scorer_version"] == scorer_version]
    elif len(versions) > 1:
        raise score_quota.VersionConflict(f"l2-head rows carry versions {versions}; pin one for selection")
    else:
        sel = rows
    if rubric_dim is not None:
        allowed = set().union(*map(set, _RUBRIC_DIMS.values()))
        if rubric_dim not in allowed:
            raise HeadScoreError(f"rubric_dim {rubric_dim!r} is not a rubric dimension")
    groups = {}
    # one document at most once per (group, pinned version): duplicates break quota
    # conservation exactly as in score_quota.pin_groups -- loud, never silently double-keep.
    seen = set()
    for r in sel:
        if rubric_dim is not None and rubric_dim not in r["dims"]:
            raise HeadScoreError(f"row {r['doc_id']} ({r['rubric_kind']}) lacks {rubric_dim}")
        key = (r["domain"], r["lang"]) if by_lang else (r["domain"], None)
        identity = (key, r["doc_id"])
        if identity in seen:
            raise HeadScoreError(
                f"duplicate doc_id {r['doc_id']!r} in group {key} for l2-head/"
                f"{r['scorer_version']}; reconcile the head-score shards (one score per doc "
                "per pinned version) before selecting a quota"
            )
        seen.add(identity)
        groups.setdefault(key, []).append((r["doc_id"], _row_value(r, rubric_dim)))
    return groups, (scorer_version or versions[0])


def _selftest():
    import tempfile

    def row(d, domain="code_py_starcoder_dc", kind="code", val=4.0, version="h1", shard=None, dims=None):
        return {
            "doc_id": d,
            "domain": domain,
            "lang": "en",
            "scorer_name": SCORER,
            "scorer_version": version,
            "ts": "2026-09-16T00:00:00Z",
            "rubric_kind": kind,
            "model": "bge-m3+head",
            "source_shard": shard,
            "src_sha": "a" * 64,
            "dim_scale": DIM_SCALE,
            "dims": dims
            or {
                "content_quality": val,
                "factual_correctness": val - 1,
                "complexity": val - 2,
                "educational_or_code_value": val + 0.5,
            },
        }

    good = row("d1")
    validate_row(good)
    # continuous values are kept as-is, including a value OUTSIDE 1..5 (not clamped/rounded)
    over = row(
        "d2",
        dims={
            "content_quality": 5.7,
            "factual_correctness": 0.3,
            "complexity": 2.0,
            "educational_or_code_value": 4.0,
        },
    )
    validate_row(over)
    assert over["dims"]["content_quality"] == 5.7 and over["dims"]["factual_correctness"] == 0.3
    # natural_language kind shares the same four keys
    validate_row(row("d3", domain="en_c4_stage2_dc", kind="natural_language"))

    def expect_bad(mut, label):
        r = json.loads(json.dumps(good))
        mut(r)
        try:
            validate_row(r)
        except HeadScoreError:
            return
        raise AssertionError(f"accepted bad row: {label}")

    expect_bad(lambda r: r.update(ts="2026-09-16 00:00"), "ts not UTC Z")
    expect_bad(lambda r: r.update(dim_scale="other"), "dim scale")
    expect_bad(lambda r: r.update(rubric_kind="math"), "unknown kind")
    expect_bad(lambda r: r["dims"].pop("complexity"), "missing dim")
    expect_bad(lambda r: r["dims"].update(complexity=float("nan")), "nan dim")
    expect_bad(lambda r: r["dims"].update(complexity=True), "bool dim")
    expect_bad(lambda r: r.update(src_sha="deadbeef"), "bad src_sha")

    # adapter keeps the float signal; mean and single-dim ordering both select via quota
    rows = [row(f"d{i:02d}", val=1.0 + i * 0.13) for i in range(20)]
    g, ver = build_groups(rows, "h1")
    assert ver == "h1" and all(isinstance(v, float) for vs in g.values() for _, v in vs)
    decs = score_quota.select_thresholds(g, {"code_py_starcoder_dc": 0.5})
    dec = decs[("code_py_starcoder_dc", "en")]
    assert dec["keep"] == 10 and dec["total"] == 20
    # mean ordering: the ten highest-mean docs survive, ties broken by doc_id
    means = {r["doc_id"]: sum(r["dims"].values()) / 4 for r in rows}
    top = set(sorted(means, key=lambda d: (-means[d], d))[:10])
    assert set(dec["kept_doc_ids"]) == top
    # single-dim ordering uses that dim directly
    g2, _ = build_groups(rows, "h1", rubric_dim="complexity")
    vals = {r["doc_id"]: r["dims"]["complexity"] for r in rows}
    d2 = score_quota.select_thresholds(g2, {"code_py_starcoder_dc": 0.25})[("code_py_starcoder_dc", "en")]
    assert set(d2["kept_doc_ids"]) == set(sorted(vals, key=lambda d: (-vals[d], d))[:5])

    # unpinned multi-version selection refuses (no blended snapshots)
    multi = rows + [row("dx", version="h2")]
    try:
        build_groups(multi)
    except score_quota.VersionConflict:
        pass
    else:
        raise AssertionError("mixed head versions must raise")

    # a duplicate doc_id in one pinned group refuses (would double-retain under a quota)
    try:
        build_groups(rows + [dict(rows[0])], "h1")
    except HeadScoreError:
        pass
    else:
        raise AssertionError("duplicate l2-head doc_id must raise")

    # append/load round trip
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "preds.jsonl")
        assert append_rows(p, [good, over]) == 2
        loaded = load_rows(p)
        assert loaded[1]["dims"]["content_quality"] == 5.7 and len(loaded) == 2

        # all-or-nothing: a bad row in the MIDDLE of a batch must not leave its good prefix on
        # disk (the old per-row validate-then-write committed g before reaching the bad row).
        bad = dict(good)
        bad["dims"] = dict(good["dims"])
        bad["dims"]["content_quality"] = float("nan")
        try:
            append_rows(p, [row("g_mid"), bad])
        except HeadScoreError:
            pass
        else:
            raise AssertionError("bad row mid-batch must raise")
        assert len(load_rows(p)) == 2, "failed batch partially appended a good prefix"
        # a row that passes schema validation but is not JSON-serializable must ALSO fail before
        # any byte lands (serialization is front-loaded, not deferred into the write loop).
        unser = {**row("g_unser"), "extra": {"a", "b"}}
        try:
            append_rows(p, [row("g_pre"), unser])
        except TypeError:
            pass
        else:
            raise AssertionError("unserializable row must raise before writing")
        assert len(load_rows(p)) == 2, "unserializable batch partially appended"
        # because the failed batches wrote nothing, a clean retry appends exactly once (no dupes)
        assert append_rows(p, [row("g3")]) == 1 and len(load_rows(p)) == 3
        # an empty batch writes nothing and reports 0 without touching the file
        assert append_rows(p, []) == 0 and len(load_rows(p)) == 3
        # durability: a successful append fsyncs the data fd before returning (os is shared, so
        # patching os.fsync intercepts the call append_rows makes)
        real_fsync, calls = os.fsync, []
        os.fsync = lambda fd: (calls.append(fd), real_fsync(fd))[1]
        try:
            append_rows(p, [row("g4")])
        finally:
            os.fsync = real_fsync
        assert calls, "append_rows must fsync before close"
        assert len(load_rows(p)) == 4

    print(
        "l2_head_scores selftest OK: float dims unquantized (out-of-range kept), schema "
        "rejections, mean/single-dim adapter through score_quota, version conflict, round-trip, "
        "all-or-nothing append (no partial prefix/double-write), fsync"
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("append")
    p.add_argument("path")
    p.add_argument("rows_json")
    a = ap.parse_args(argv)
    if a.selftest:
        _selftest()
        return
    if a.cmd == "append":
        with open(a.rows_json, encoding="utf-8") as f:
            rows = json.load(f)
        print(f"appended {append_rows(a.path, rows)} rows -> {a.path}")
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
