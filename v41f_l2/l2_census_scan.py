"""Census-scan the corpus with the frozen bge-m3 encoder + trained L2 quality head.

The production half of the funnel: where training (v41f_l2/train_l2.py) reads 66's labelled
ledger and fits the head, THIS reads every corpus shard and writes the head's CONTINUOUS
4-dim quality scores so the per-domain quota selector (datagen/score_quota.py via de's
adapter datagen/l2_head_scores.py, PR #402) can rank and cut tens of millions of docs
cheaply (~131 docs/s/card forward, facts dq.l2_*_0916).

What is scored
--------------
The WHOLE document, not 1024-token pieces. bge-m3's hard context is 8192 and the probe
measured ~0.13% of real docs longer than that, so each doc is tokenized with truncation at
8192 and passed once; doc_id is content_doc_id(whole text). The 1024 chunker in
l2_label_pool_build exists for the L3 teacher's context window and is deliberately NOT used
here. A doc that hits the 8192 cap is counted (over_ctx) because a nonzero total means the
length distribution moved and the head silently saw a truncated doc.

Output: continuous, never int-quantized
---------------------------------------
Rows are written through datagen/l2_head_scores.py (the canonical schema, PR #402): four
finite floats on the native rubric_1_5 head scale, allowed to fall slightly outside 1..5,
never clamped/rounded. They do NOT enter score_ledger.rubric_dims -- that column is the
teacher's int 1..5 labels, a different signal. l2_head_scores.build_groups adapts these rows
straight into score_quota.select_thresholds / export_review_sample. An int 1..5 view is a
downstream display-only concern and is intentionally not stored.

Sharding / resume
-----------------
One output file per (domain, shard). Workers take a disjoint strided subset
(shards[r::num_shards]) -> near-linear multi-card scaling. A finished output is skipped on
re-run (idempotent); a shard is staged to .tmp and atomically renamed, so a crash leaves no
partial/duplicate rows. Within a shard, content doc_ids de-duplicate.

Throughput follows the probe: CPU tokenize is pipelined ahead of GPU forward via a prefetch
worker; torch/HF is imported lazily, so the selftest (stub predictor) never touches a GPU.

    python3 v41f_l2/l2_census_scan.py --selftest
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
# canonical continuous-score schema + quota adapter (de, PR #402); single source of the
# scorer name, dim scale, row validation and append/load. Not redefined here.
from datagen.l2_head_scores import DIM_SCALE, SCORER, append_rows, load_rows  # noqa: E402
from datagen.l3_rubric import _DIMS as RUBRIC_DIMS  # noqa: E402
from datagen.score_ledger import (  # noqa: E402
    assert_doc_matches_content,
    content_doc_id,
)

MAX_CTX = 8192  # bge-m3 hard context; a doc tokenizing to this cap is counted, expected ~0


@dataclass
class ScanConfig:
    scorer_version: str
    model_id: str = "BAAI/bge-m3"
    domain_kind: dict | None = None  # domain -> rubric_kind; heuristic default below
    token_budget: int = 32768  # tokens per forward pack (probe throughput shape)
    lang: str | None = None

    def kind_of(self, domain: str) -> str:
        if self.domain_kind:
            return self.domain_kind[domain]
        return "code" if ("code" in domain or "starcoder" in domain) else "natural_language"


# --------------------------------------------------------------------------------------
# pure, GPU-free pieces (the selftest drives these with a stub predictor)


def enumerate_shards(domain_globs: dict) -> list[tuple[str, str]]:
    """domain -> glob -> sorted [(domain, shard_path)]. Deterministic order so shard
    assignment is stable across workers and re-runs."""
    out = []
    for domain, pattern in sorted(domain_globs.items()):
        for path in sorted(glob.glob(pattern)):
            out.append((domain, path))
    return out


def assign_shards(shards: list, shard_idx: int, num_shards: int) -> list:
    """Strided assignment shards[r::num_shards]: independent, disjoint, union-complete
    subsets for near-linear multi-card scans. Raises on a bad split."""
    if num_shards <= 0 or not (0 <= shard_idx < num_shards):
        raise ValueError(f"shard {shard_idx}/{num_shards} invalid")
    return shards[shard_idx::num_shards]


def shard_output_path(out_dir: str, domain: str, shard_path: str) -> Path:
    stem = Path(shard_path).stem
    safe_dom = domain.replace("/", "_")
    return Path(out_dir) / f"{safe_dom}__{stem}.{SCORER}.jsonl"


def marker_path(out_path: Path) -> Path:
    """The completion/fingerprint sidecar for one scored shard (.jsonl -> .done.json)."""
    return out_path.with_suffix(".done.json")


def expected_marker(cfg: ScanConfig, head_fp: str) -> dict:
    """The identity a completed shard must match to be safely skipped. If ANY field changes
    (head retrained, version bumped, model swapped), an old output is NOT the current result
    and resume must refuse to treat it as done."""
    return {
        "scorer_name": SCORER,
        "scorer_version": cfg.scorer_version,
        "model": cfg.model_id,
        "dim_scale": DIM_SCALE,
        "head_fingerprint": head_fp,
    }


class StaleShardError(RuntimeError):
    """An output shard exists but was produced by a different head/version/model. Resuming
    over it would silently serve stale predictions; the caller must pass --force to rescan."""


def _check_resume(out_path: Path, cfg: ScanConfig, head_fp: str, force: bool) -> bool:
    """True if the shard is complete AND identical to the current fingerprint (skip it).
    Raises StaleShardError on a fingerprint mismatch unless force. A data file without its
    marker (crash before the rename of the marker, or an older scanner) is treated as
    incomplete and rescanned, not skipped."""
    if not out_path.exists():
        return False
    if force:
        # --force means unconditionally rescan, even a current, fingerprint-matched output
        return False
    mp = marker_path(out_path)
    if not mp.exists():
        raise StaleShardError(
            f"{out_path} exists but has no completion marker; its head/version is unknown. "
            "Re-run with --force to rescan it instead of trusting unverifiable output."
        )
    done = json.loads(mp.read_text())
    want = expected_marker(cfg, head_fp)
    changed = {k: {"have": done.get(k), "want": want[k]} for k in want if done.get(k) != want[k]}
    if changed:
        raise StaleShardError(
            f"{out_path} was scored with a different identity: {json.dumps(changed)}. "
            "Refusing to reuse stale output; pass --force to rescan and overwrite."
        )
    return True


def _write_marker(out_path: Path, cfg: ScanConfig, head_fp: str, n_rows: int, over_ctx_count: int = 0):
    # over_ctx_count is a per-shard OUTPUT statistic (docs whose bge-m3 input hit the 8192
    # cap and were truncated), persisted here so truncation is traceable from the marker
    # without scraping stdout. It is deliberately NOT in expected_marker's identity fields:
    # it describes what was scored, not which head/version/model scored it.
    payload = {
        **expected_marker(cfg, head_fp),
        "rows": n_rows,
        "over_ctx_count": over_ctx_count,
        "over_ctx_ctx": MAX_CTX,
        "ts": _utcnow(),
    }
    mtmp = marker_path(out_path).with_suffix(".done.json.tmp")
    with open(mtmp, "w") as fh:
        json.dump(payload, fh, sort_keys=True)
    _durably_rename(mtmp, marker_path(out_path))


def read_corpus_docs(shard_path: str, counters: dict):
    """Stream (content, src_sha) for every USABLE JSONL row, accounting for every input row.

    Nothing is dropped silently. Counters:
      input_rows        every non-blank physical line attempted,
      bad_content_rows  lines that are not JSON, are not an object, lack "content", carry a
                        non-string content, or an empty/whitespace string.
    A blank separator line is ignored without counting. All-bad input surfaces upstream as a
    shard with zero valid rows, which is refused rather than silently completed."""
    with open(shard_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            counters["input_rows"] = counters.get("input_rows", 0) + 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                counters["bad_content_rows"] = counters.get("bad_content_rows", 0) + 1
                continue
            if not isinstance(row, dict):
                counters["bad_content_rows"] = counters.get("bad_content_rows", 0) + 1
                continue
            content = row.get("content")
            if not isinstance(content, str) or not content.strip():
                counters["bad_content_rows"] = counters.get("bad_content_rows", 0) + 1
                continue
            yield content, row.get("src_sha")


def build_row(content, vec, cfg: ScanConfig, domain, shard_path, src_sha=None, enc_over_ctx=False):
    """One whole-doc 4-vector -> a canonical head-score row (continuous, unquantized).
    doc_id is the whole-content hash, re-checked against the bytes.

    enc_over_ctx=True marks a doc whose bge-m3 input tokenized to the 8192 cap and was
    truncated before the head saw it (encoder-side, distinct from the teacher's 6000-char
    label truncation in datagen/l2_dataset.py). It is an extra schema key; validate_row
    checks required fields and passes unknown keys through, so the canonical reader keeps
    working while the flag is traceable per row."""
    if len(vec) != len(RUBRIC_DIMS):
        raise ValueError(f"predictor returned {len(vec)} dims, expected {len(RUBRIC_DIMS)}")
    doc_id = content_doc_id(content)
    assert_doc_matches_content(doc_id, content)
    return {
        "doc_id": doc_id,
        "domain": domain,
        "lang": cfg.lang or "en",
        "scorer_name": SCORER,
        "scorer_version": cfg.scorer_version,
        "ts": _utcnow(),
        "rubric_kind": cfg.kind_of(domain),
        "model": cfg.model_id,
        "source_shard": shard_path,
        "src_sha": src_sha,
        "dim_scale": DIM_SCALE,
        "enc_over_ctx": bool(enc_over_ctx),
        "dims": {name: float(x) for name, x in zip(RUBRIC_DIMS, vec, strict=True)},
    }


def _utcnow() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _fsync_file(path):
    """Flush file data to disk so a later rename does not expose a non-durable file."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_dir(dir_path):
    """fsync a directory so a file creation/rename inside it is durable (POSIX requirement)."""
    fd = os.open(dir_path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _durably_rename(src: Path, dst: Path):
    """fsync the source, atomically rename to dst, then fsync the parent directory."""
    _fsync_file(str(src))
    os.replace(src, dst)
    _fsync_dir(str(dst.parent))


class EmptyShardError(RuntimeError):
    """A shard produced ZERO valid score rows. Marking it complete would make resume skip it
    forever, silently giving an all-bad/empty domain zero scores; fail loud instead."""


def scan_shards(domain, paths, out_dir, predict, cfg: ScanConfig, counters, *, force=False):
    """Score a worker's assigned shards, one validated output file + fingerprint marker per
    shard.

    Idempotent and version-safe: a shard is skipped only when its data file AND .done.json
    marker exist and the marker's scorer/version/model/head fingerprint match THIS run. A
    changed head or version makes the existing output stale -> StaleShardError (use --force).
    Each shard stages its data to .tmp and atomically renames, then writes the marker, so a
    crash never leaves partial/duplicate rows or a claim of completion without scores.
    Dedup is DOMAIN-level across this worker's shards: one content-id set is shared, so
    identical content appearing in two shards is scored once and counted in dup_docs (doc_id
    is a content hash, so the same bytes anywhere are the same doc). Mutates the shared
    counters dict; returns written.

    Cross-worker duplicates (two cards scanning different files that hold identical content)
    cannot share an in-memory set; audit them afterwards with find_domain_duplicates over the
    shared out_dir."""
    head_fp = head_fingerprint(predict)
    written = 0
    domain_seen: set[str] = set()
    for shard_path in paths:
        out_path = shard_output_path(out_dir, domain, shard_path)
        if _check_resume(out_path, cfg, head_fp, force):
            counters["skipped_shards"] += 1
            continue
        rows, stats = _score_one_shard(domain, shard_path, predict, cfg, counters, domain_seen)
        if not rows:
            # 0 kept splits two ways. dup>0: every USABLE row was normal cross-shard
            # dedup (#420's shared domain_seen) — the content is already scored in an earlier
            # shard, so publishing nothing is correct; skip, do not mark this file complete.
            # dup==0: the shard was empty/blank or every input row was bad, so nothing covers
            # its content anywhere — refuse instead of stamping a zero-row shard done forever.
            if stats["dup"] > 0:
                counters["dup_only_shards"] = counters.get("dup_only_shards", 0) + 1
                continue
            raise EmptyShardError(
                f"{shard_path}: 0 valid scored rows "
                f"(input_rows={stats['input']}, bad_content_rows={stats['bad']}, "
                f"dup_dropped={stats['dup']}). Not publishing a file or completion marker; "
                "fix the shard/glob before trusting this domain."
            )
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        if tmp.exists():
            tmp.unlink()
        # append_rows validates each row against the canonical schema before any byte lands;
        # publish + marker are both fsync/rename/fsync-dir durable
        append_rows(str(tmp), rows)
        _durably_rename(tmp, out_path)
        _write_marker(out_path, cfg, head_fp, len(rows), over_ctx_count=stats["over_ctx"])
        written += len(rows)
        counters["written_shards"] += 1
    return written


def head_fingerprint(predict) -> str:
    """A stable identity for the scoring head. A real predictor (HeadPredictor) hashes its
    loaded head state_dict; a stub returns a constant. Different weights -> different fp ->
    resume refuses to reuse the old output."""
    fp = getattr(predict, "fingerprint", None)
    return fp() if callable(fp) else "stub-head"


def find_domain_duplicates(out_dir, domain):
    """Read every scored file for a domain and return {doc_id: [output files]} for docs
    appearing in MORE than one shard file. Works across workers because all shard outputs
    land in one out_dir with domain-prefixed names. doc_id is a content hash, so a repeated id
    is byte-identical content double-counted across shards — a census must surface it."""
    prefix = domain.replace("/", "_")
    files = sorted(glob.glob(os.path.join(out_dir, f"{prefix}__*.{SCORER}.jsonl")))
    owners: dict[str, set] = {}
    for fp in files:
        for r in load_rows(fp):
            owners.setdefault(r["doc_id"], set()).add(fp)
    return {doc_id: sorted(fs) for doc_id, fs in owners.items() if len(fs) > 1}


DOC_BATCH = 256  # unique docs per predictor call; bounds text held in memory


def _predict_vecs(predict, texts):
    """Call a predictor and return (vecs, over_ctx_flags) aligned to `texts`.

    The production HeadPredictor returns the tuple (flags mark per-doc 8192 truncation). A
    predictor/selftest stub that returns only vectors is accepted with all-False flags, so
    the scoring path tolerates a plain callable but a real truncation flag is never faked."""
    out = predict(texts)
    if isinstance(out, tuple) and len(out) == 2:
        vecs, flags = out
    else:
        vecs, flags = out, [False] * len(texts)
    if len(vecs) != len(texts):
        raise ValueError(f"predictor returned {len(vecs)} vectors for {len(texts)} docs")
    if len(flags) != len(texts):
        raise ValueError(f"predictor returned {len(flags)} over-ctx flags for {len(texts)} docs")
    return vecs, flags


def _score_one_shard(domain, shard_path, predict, cfg, counters, domain_seen):
    # Stream the shard in bounded DOC_BATCH micro-batches: a shard can be GBs, so we never
    # hold all its text. Within a batch the predictor packs to the GPU token budget (the 131
    # docs/s shape). domain_seen is the cross-shard content-id set for this worker.
    # Returns (rows, this-shard stats); the stats distinguish "every usable row was a normal
    # cross-shard duplicate" from "the shard was empty or all bad", which scan_shards decides
    # on (skip vs refuse).
    rows = []
    in0 = counters.get("input_rows", 0)
    bad0 = counters.get("bad_content_rows", 0)
    dup0 = counters["dup_docs"]
    over0 = counters.get("over_ctx", 0)

    def flush(batch):
        if not batch:
            return
        vecs, flags = _predict_vecs(predict, [c for c, _ in batch])
        for (content, src_sha), vec, over in zip(batch, vecs, flags, strict=True):
            rows.append(build_row(content, vec, cfg, domain, shard_path, src_sha=src_sha, enc_over_ctx=over))
            counters["docs"] += 1
            if over:
                counters["over_ctx"] = counters.get("over_ctx", 0) + 1

    batch = []
    for content, src_sha in read_corpus_docs(shard_path, counters):
        cid = content_doc_id(content)
        if cid in domain_seen:
            counters["dup_docs"] += 1
            continue
        domain_seen.add(cid)
        batch.append((content, src_sha))
        if len(batch) >= DOC_BATCH:
            flush(batch)
            batch = []
    flush(batch)
    stats = {
        "input": counters.get("input_rows", 0) - in0,
        "bad": counters.get("bad_content_rows", 0) - bad0,
        "dup": counters["dup_docs"] - dup0,
        "kept": len(rows),
        "over_ctx": counters.get("over_ctx", 0) - over0,
    }
    return rows, stats


# --------------------------------------------------------------------------------------
# GPU / HF edge (lazily imported; the selftest never constructs this)


class HeadPredictor:
    """Frozen bge-m3 + trained QualityHead over WHOLE docs truncated at 8192, packed into
    token-budget batches. Encoder pooling reuses L2QualityModel so the scanner cannot drift
    from training's pooler. A production run wraps this in a tokenize-ahead prefetch worker;
    the per-batch call boundary below is the overlap seam."""

    def __init__(self, head_ckpt: str, cfg: ScanConfig, device: str = "cuda"):
        import torch
        from transformers import AutoModel, AutoTokenizer

        from v41f_l2.l2_quality_head import L2Config, L2QualityModel, load_head_state

        self.torch = torch
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(cfg.model_id)
        enc = AutoModel.from_pretrained(cfg.model_id, torch_dtype=torch.float16)
        head_state, head_cfg = load_head_state(head_ckpt)
        model = L2QualityModel(enc, L2Config(**head_cfg) if head_cfg else L2Config(), pooling="cls")
        model.head.load_state_dict(head_state)
        self.model = model.to(device).eval()
        self.token_budget = cfg.token_budget
        self.over_ctx = 0

    def fingerprint(self) -> str:
        """sha256 over the head parameters; a retrained head yields a different fp so resume
        cannot treat old predictions as the new head's output."""
        import hashlib

        h = hashlib.sha256()
        for k in sorted(self.model.head.state_dict()):
            t = self.model.head.state_dict()[k].float().cpu().contiguous()
            h.update(k.encode())
            h.update(t.numpy().tobytes())
        return f"headsha256:{h.hexdigest()[:16]}"

    def _encode_batch(self, texts):
        torch = self.torch
        enc = self.tok(texts, padding=True, truncation=True, max_length=MAX_CTX, return_tensors="pt")
        # a doc at the cap was truncated; probe measured ~0, so a nonzero count is a signal.
        # Return per-doc flags (not just the running total) so the flag rides on the row.
        over = (enc["attention_mask"].sum(1) >= MAX_CTX).tolist()
        self.over_ctx += int(sum(over))
        with torch.no_grad():
            pred = self.model(enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device))
        return pred.float().cpu().tolist(), [bool(x) for x in over]

    def __call__(self, texts):
        # whole docs, greedy token-budget packs (length-sorting upstream reduces padding in a
        # production run; the call boundary stays one batch in / one (vec,flag)-per-doc out).
        out, flags, cur, cur_tok = [], [], [], 0
        for t in texts:
            ntok = len(self.tok(t, add_special_tokens=True)["input_ids"])
            if cur and cur_tok + min(ntok, MAX_CTX) > self.token_budget:
                v, f = self._encode_batch(cur)
                out.extend(v)
                flags.extend(f)
                cur, cur_tok = [], 0
            cur.append(t)
            cur_tok += min(ntok, MAX_CTX)
        if cur:
            v, f = self._encode_batch(cur)
            out.extend(v)
            flags.extend(f)
        return out, flags


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out-dir", default="runs/l2_census")
    ap.add_argument("--head-ckpt", help="trained QualityHead state_dict")
    ap.add_argument("--scorer-version")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument(
        "--domains",
        nargs="*",
        default=["code_py_starcoder_dc", "en_c4_stage2_dc", "code_ultra_l2_dc"],
    )
    ap.add_argument("--corpus-root", default="/work/aupai/data/corpus")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return
    if not args.scorer_version:
        ap.error("--scorer-version is required for a real scan")
    if not args.head_ckpt:
        ap.error("--head-ckpt is required for a real scan")

    globs = {d: os.path.join(args.corpus_root, d, "*.jsonl") for d in args.domains}
    all_shards = enumerate_shards(globs)
    if not all_shards:
        # a glob typo / wrong corpus root / rebuilt-but-empty domain would otherwise finish
        # with zero scores and look successful; that is misconfiguration, not an empty census.
        raise SystemExit(f"no corpus shards matched {globs}; check --corpus-root/--domains")
    shards = assign_shards(all_shards, args.shard, args.num_shards)
    if not shards:
        raise SystemExit(f"worker --shard {args.shard}/--num-shards {args.num_shards} was assigned 0 shards")
    cfg = ScanConfig(scorer_version=args.scorer_version)
    predict = HeadPredictor(args.head_ckpt, cfg)
    by_domain: dict[str, list] = {}
    for domain, path in shards:
        by_domain.setdefault(domain, []).append(path)
    counters = {
        "docs": 0,
        "dup_docs": 0,
        "written_shards": 0,
        "skipped_shards": 0,
        "dup_only_shards": 0,
        "over_ctx": 0,
        "input_rows": 0,
        "bad_content_rows": 0,
    }
    for domain, paths in sorted(by_domain.items()):
        scan_shards(domain, paths, args.out_dir, predict, cfg, counters, force=args.force)
    # probe measured ~0 docs at 8192; nonzero means the length distribution moved and docs
    # were truncated. The durable sources are per-row enc_over_ctx + each marker's
    # over_ctx_count; this line is only the run's human-readable closing summary.
    print(f"{counters} over_ctx(={MAX_CTX})={counters['over_ctx']}", flush=True)
    # cross-SHARD / cross-worker audit: this worker only dedups the shards it scanned. After
    # all workers write the shared out_dir, identical content in two shard FILES (e.g. two
    # cards scanning files that hold the same bytes) is byte-identical double counting. A
    # nonzero census duplicate count is a data-integrity failure, so the scan exits nonzero.
    total_xdup = 0
    for domain in sorted(by_domain):
        xdups = find_domain_duplicates(args.out_dir, domain)
        if xdups:
            total_xdup += len(xdups)
            sample = next(iter(xdups))
            print(
                f"CROSS-SHARD DUP {domain}: {len(xdups)} doc_id(s) in >1 shard file; "
                f"e.g. {sample} -> {xdups[sample]}",
                flush=True,
            )
    if total_xdup:
        raise SystemExit(
            f"{total_xdup} cross-shard duplicate doc_id(s) found in {args.out_dir}; the census "
            "double-counted them. Re-dedup the corpus or shard assignment before selecting."
        )
    # bad content rows must never be a silent loss; a nonzero count fails the scan so an
    # upstream parse/encoding problem cannot quietly shrink a domain.
    if counters["bad_content_rows"]:
        raise SystemExit(
            f"{counters['bad_content_rows']} unparseable/empty/missing-content rows skipped "
            f"of {counters['input_rows']} input rows; fix the corpus before trusting the census"
        )


def _selftest():
    import tempfile

    import datagen.l2_head_scores as hs

    dims = tuple(RUBRIC_DIMS)
    assert len(dims) == 4
    assert SCORER == "l2-head" and DIM_SCALE == "rubric_1_5"

    # shard assignment: disjoint, union-complete, deterministic; out-of-range raises
    shards = [(f"d{i % 2}", f"/x/shard{i}.jsonl") for i in range(10)]
    a, b = assign_shards(shards, 0, 2), assign_shards(shards, 1, 2)
    assert sorted(a + b) == sorted(shards) and set(a).isdisjoint(b)
    assert assign_shards(shards, 0, 2) == a
    for bad in (lambda: assign_shards(shards, 2, 2), lambda: assign_shards(shards, 0, 0)):
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError("bad shard split must raise")

    with tempfile.TemporaryDirectory() as td:

        def write_shard(name, docs):
            p = os.path.join(td, name)
            with open(p, "w") as fh:
                for c in docs:
                    fh.write(json.dumps({"content": c}) + "\n")
            return p

        s0 = write_shard("s0.jsonl", ["print(1)\n", "x = 2\n", "print(1)\n"])  # in-shard dup
        s1 = write_shard("s1.jsonl", ["def f(): return 3\n", "class C: pass\n"])

        # stub predictor: continuous vectors incl values OUTSIDE 1..5; must be preserved as
        # floats (no clamp/round) and stay separable for ranking.
        def stub_predict(texts):
            table = {
                "print(1)": [5.7, 2.0, 1.0, 4.0],  # out of range high, kept
                "x = 2": [0.3, 3.0, 2.0, 4.0],  # out of range low, kept
                "def f(): return 3": [4.0, 4.0, 4.0, 4.0],
                "class C: pass": [1.0, 1.0, 1.0, 1.0],
            }
            return [table[t.strip()] for t in texts]

        cfg = ScanConfig(scorer_version="selftest-r1", lang="en")
        counters = {"docs": 0, "dup_docs": 0, "written_shards": 0, "skipped_shards": 0}

        # first pass: s0 has 3 lines but one duplicate -> 2 unique docs
        n = scan_shards("code_py", [s0, s1], td, stub_predict, cfg, counters)
        assert n == 4, n
        assert counters["dup_docs"] == 1 and counters["written_shards"] == 2, counters

        out_files = sorted(glob.glob(os.path.join(td, "*.l2-head.jsonl")))
        assert len(out_files) == 2, out_files

        # every row validates against the CANONICAL continuous schema
        rows = []
        for f in out_files:
            rows.extend(load_rows(f))
        assert len(rows) == 4
        for r in rows:
            assert set(r["dims"]) == set(dims)
            assert all(isinstance(v, float) for v in r["dims"].values())
            assert r["dim_scale"] == DIM_SCALE and r["scorer_name"] == SCORER
            assert r["rubric_kind"] == "code" and r["model"] == cfg.model_id

        # continuous values stored UNQUANTIZED, including outside 1..5
        hi = next(r for r in rows if r["dims"][dims[0]] > 5)
        lo = next(r for r in rows if r["dims"][dims[0]] < 1)
        assert hi["dims"][dims[0]] == 5.7 and lo["dims"][dims[0]] == 0.3

        # ranking is separable on the continuous mean (the selector's ordering input):
        # build_groups -> score_quota ranking must put the 4.0 doc above the 1.0 doc.
        from datagen import score_quota

        groups, version = hs.build_groups(rows, scorer_version="selftest-r1")
        assert version == "selftest-r1"
        ranked = sorted(groups[("code_py", "en")], key=lambda dv: (-dv[1], dv[0]))
        top_id, top_val = ranked[0]
        bot_id, bot_val = ranked[-1]
        assert top_val > bot_val and top_id != bot_id
        # exact equivalence: adapter value for an all-4.0 doc is the same float as direct mean
        four = next(r for r in rows if all(v == 4.0 for v in r["dims"].values()))
        assert dict(groups[("code_py", "en")])[four["doc_id"]] == 4.0
        # the canonical guard refuses a multi-version selection
        mixed = rows + [{**rows[0], "scorer_version": "other"}]
        try:
            hs.build_groups(mixed)
        except score_quota.VersionConflict:
            pass
        else:
            raise AssertionError("mixed scorer_version must raise VersionConflict")

        # idempotent resume: re-run skips both finished shards, writes nothing, no dup rows
        c2 = {"docs": 0, "dup_docs": 0, "written_shards": 0, "skipped_shards": 0}
        n2 = scan_shards("code_py", [s0, s1], td, stub_predict, cfg, c2)
        assert n2 == 0 and c2["skipped_shards"] == 2 and c2["written_shards"] == 0
        assert len(load_rows(out_files[0])) + len(load_rows(out_files[1])) == 4
        # every completed shard carries a fingerprint marker naming this head/version/model
        for of in out_files:
            mk = json.loads(marker_path(Path(of)).read_text())
            assert mk["head_fingerprint"] == "stub-head"
            assert mk["scorer_version"] == cfg.scorer_version and mk["model"] == cfg.model_id
            assert mk["dim_scale"] == DIM_SCALE and mk["rows"] >= 1

        # STALE RESUME: tamper the marker to a different head fingerprint -> reuse refused
        m0 = marker_path(shard_output_path(td, "code_py", s0))
        good_marker = json.loads(m0.read_text())
        tampered = {**good_marker, "head_fingerprint": "headsha256:deadbeefdeadbe"}
        m0.write_text(json.dumps(tampered))
        try:
            scan_shards(
                "code_py",
                [s0],
                td,
                stub_predict,
                cfg,
                {"docs": 0, "dup_docs": 0, "written_shards": 0, "skipped_shards": 0},
            )
        except StaleShardError:
            pass
        else:
            raise AssertionError("a different head fingerprint on resume must raise")
        # and a changed scorer_version in the run config is equally stale
        m0.write_text(json.dumps(good_marker))
        cfg2 = ScanConfig(scorer_version="selftest-r2", lang="en")
        try:
            scan_shards(
                "code_py",
                [s0],
                td,
                stub_predict,
                cfg2,
                {"docs": 0, "dup_docs": 0, "written_shards": 0, "skipped_shards": 0},
            )
        except StaleShardError:
            pass
        else:
            raise AssertionError("a different scorer_version on resume must raise")
        # --force overrides the staleness refusal and rescans (re-writing the new marker)
        c_f = {"docs": 0, "dup_docs": 0, "written_shards": 0, "skipped_shards": 0}
        scan_shards("code_py", [s0], td, stub_predict, cfg2, c_f, force=True)
        assert c_f["written_shards"] == 1
        assert json.loads(m0.read_text())["scorer_version"] == "selftest-r2"
        m0.write_text(json.dumps(good_marker))  # restore for the forced check below

        # an output with NO marker (older scanner / crash) is not silently trusted
        m1 = marker_path(shard_output_path(td, "code_py", s1))
        m1.unlink()
        try:
            scan_shards(
                "code_py",
                [s1],
                td,
                stub_predict,
                cfg,
                {"docs": 0, "dup_docs": 0, "written_shards": 0, "skipped_shards": 0},
            )
        except StaleShardError:
            pass
        else:
            raise AssertionError("markerless existing output must not be skipped as complete")

        # a forced rescan still validates (crash/.tmp safety exercised via overwrite)
        c3 = {"docs": 0, "dup_docs": 0, "written_shards": 0, "skipped_shards": 0}
        scan_shards("code_py", [s0], td, stub_predict, cfg, c3, force=True)
        assert c3["written_shards"] == 1 and len(load_rows(shard_output_path(td, "code_py", s0))) == 2

    # DOMAIN-LEVEL dedup across shards (audit fix #4). Same content in two files scanned by ONE
    # worker (shared set) is scored once and counted; the two output files do not both hold it.
    with tempfile.TemporaryDirectory() as dd:

        def any_predict(texts):
            return [[3.0, 3.0, 3.0, 3.0] for _ in texts]

        a = os.path.join(dd, "a.jsonl")
        b = os.path.join(dd, "b.jsonl")
        for p, docs in (
            (a, ["shared content body\n", "alpha only\n"]),
            (b, ["shared content body\n", "beta only\n"]),
        ):
            with open(p, "w") as fh:
                fh.writelines(json.dumps({"content": c}) + "\n" for c in docs)
        dc = {"docs": 0, "dup_docs": 0, "written_shards": 0, "skipped_shards": 0}
        written = scan_shards("code_py", [a, b], dd, any_predict, cfg, dc)
        assert dc["dup_docs"] == 1, dc  # the cross-file copy hit the shared domain set
        # 3 unique docs, scored once each (shared one appears in exactly one output file)
        assert written == 3, written
        x = find_domain_duplicates(dd, "code_py")
        assert x == {}, x  # within one worker, no doc_id lands in two files

    # CROSS-WORKER case: two workers cannot share an in-memory set, so simulate two independent
    # scan_shards calls over files holding identical content; both files keep the doc and the
    # post-hoc audit MUST detect the double count.
    with tempfile.TemporaryDirectory() as dd2:

        def any_predict2(texts):
            return [[2.0, 2.0, 2.0, 2.0] for _ in texts]

        w0 = os.path.join(dd2, "w0.jsonl")
        w1 = os.path.join(dd2, "w1.jsonl")
        for p in (w0, w1):
            with open(p, "w") as fh:
                fh.write(json.dumps({"content": "same bytes on both cards\n"}) + "\n")
        scan_shards(
            "code_py",
            [w0],
            dd2,
            any_predict2,
            cfg,
            {"docs": 0, "dup_docs": 0, "written_shards": 0, "skipped_shards": 0},
        )
        scan_shards(
            "code_py",
            [w1],
            dd2,
            any_predict2,
            cfg,
            {"docs": 0, "dup_docs": 0, "written_shards": 0, "skipped_shards": 0},
        )
        xdups = find_domain_duplicates(dd2, "code_py")
        assert len(xdups) == 1 and len(next(iter(xdups.values()))) == 2, xdups

    # BAD-ROW ACCOUNTING + EMPTY-SHARD REFUSAL (hardening). Bad lines are counted, not
    # silently dropped; a zero-valid shard is neither published nor marker-completed.
    def const_pred(texts):
        return [[3.0, 3.0, 3.0, 3.0] for _ in texts]

    with tempfile.TemporaryDirectory() as bd:
        mixed = os.path.join(bd, "mixed.jsonl")
        with open(mixed, "w") as fh:
            fh.write(json.dumps({"content": "good doc here"}) + "\n")
            fh.write("\n")  # blank separator: ignored, NOT counted
            fh.write("not json at all\n")  # unparseable
            fh.write(json.dumps({"content": ""}) + "\n")  # empty
            fh.write(json.dumps({"content": 42}) + "\n")  # non-string
            fh.write(json.dumps({"nope": "x"}) + "\n")  # missing key
        hc = {
            "docs": 0,
            "dup_docs": 0,
            "written_shards": 0,
            "skipped_shards": 0,
            "input_rows": 0,
            "bad_content_rows": 0,
        }
        scan_shards("code_py", [mixed], bd, const_pred, cfg, hc)
        assert hc["input_rows"] == 5, hc  # 1 good + 4 bad; blank not counted
        assert hc["bad_content_rows"] == 4, hc
        assert hc["docs"] == 1 and hc["written_shards"] == 1

        empty = os.path.join(bd, "empty.jsonl")
        with open(empty, "w") as fh:
            fh.write(json.dumps({"content": ""}) + "\n")
            fh.write("garbage\n")
        ec = {
            "docs": 0,
            "dup_docs": 0,
            "written_shards": 0,
            "skipped_shards": 0,
            "input_rows": 0,
            "bad_content_rows": 0,
        }
        try:
            scan_shards("code_py", [empty], bd, const_pred, cfg, ec)
        except EmptyShardError:
            pass
        except Exception as e:  # bypassed guard falls through to a missing-tmp rename: still loud
            raise AssertionError(
                f"zero-valid shard must raise EmptyShardError before any publish, got {type(e).__name__}"
            ) from e
        else:
            raise AssertionError("a zero-valid shard must raise, not publish an empty result")
        assert not shard_output_path(bd, "code_py", empty).exists(), "no data published for empty shard"
        assert not marker_path(shard_output_path(bd, "code_py", empty)).exists(), "no marker for empty shard"
        assert not shard_output_path(bd, "code_py", empty).with_suffix(".tmp").exists()

        # ALL-DUPLICATE shard: two shards holding the same content are a normal #420 cross-shard
        # case. The second yields 0 kept but every usable row deduped, so scan_shards skips it
        # (counted dup_only_shards) instead of failing; its content is already in shard one.
        d1 = os.path.join(bd, "dup_a.jsonl")
        d2 = os.path.join(bd, "dup_b.jsonl")
        for dp in (d1, d2):
            with open(dp, "w") as fh:
                fh.write(json.dumps({"content": "shared across shards"}) + "\n")
        dc = {
            "docs": 0,
            "dup_docs": 0,
            "written_shards": 0,
            "skipped_shards": 0,
            "dup_only_shards": 0,
            "input_rows": 0,
            "bad_content_rows": 0,
        }
        n = scan_shards("code_py", [d1, d2], bd, const_pred, cfg, dc)
        assert n == 1, n  # one unique doc scored once
        assert dc["dup_docs"] == 1 and dc["dup_only_shards"] == 1, dc
        assert dc["written_shards"] == 1, dc
        # the all-duplicate shard publishes NOTHING and is never marked complete
        assert not shard_output_path(bd, "code_py", d2).exists()
        assert not marker_path(shard_output_path(bd, "code_py", d2)).exists()

        # ENCODER OVER-CTX: a doc at the 8192 input cap is flagged on ITS row and counted in
        # the marker, so truncation is traceable after the process exits (not stdout-only).
        long_doc = "tok " * 9000  # stands in for a doc whose bge-m3 tokenization hits 8192
        oc_path = os.path.join(bd, "overctx.jsonl")
        with open(oc_path, "w") as fh:
            fh.write(json.dumps({"content": "short one"}) + "\n")
            fh.write(json.dumps({"content": long_doc}) + "\n")
            fh.write(json.dumps({"content": "also short"}) + "\n")

        def over_pred(texts):
            # stub the tokenizer's cap decision on byte length: the long doc is "truncated".
            return [[2.0, 2.0, 2.0, 2.0] for _ in texts], [len(t) > 8192 for t in texts]

        occ = {
            "docs": 0,
            "dup_docs": 0,
            "written_shards": 0,
            "skipped_shards": 0,
            "dup_only_shards": 0,
            "over_ctx": 0,
            "input_rows": 0,
            "bad_content_rows": 0,
        }
        scan_shards("code_py", [oc_path], bd, over_pred, cfg, occ)
        assert occ["over_ctx"] == 1 and occ["docs"] == 3, occ
        out_rows = load_rows(shard_output_path(bd, "code_py", oc_path))
        assert [r["enc_over_ctx"] for r in out_rows] == [False, True, False], [
            r["enc_over_ctx"] for r in out_rows
        ]
        mp = marker_path(shard_output_path(bd, "code_py", oc_path))
        marker = json.loads(mp.read_text())
        assert marker["over_ctx_count"] == 1, marker
        assert marker["over_ctx_ctx"] == MAX_CTX, marker
        # over_ctx_count is an output statistic, NOT a resume-fingerprint identity field
        assert "over_ctx_count" not in expected_marker(cfg, head_fingerprint(over_pred))

    # DURABLE PUBLISH: file fsync before rename, parent-dir fsync after. Spy on THIS module's
    # globals (a fresh import would be a different module under `python l2_census_scan.py`).
    g = globals()
    calls = {"file": 0, "dir": 0, "replace": 0}
    of_file, of_dir, of_repl = g["_fsync_file"], g["_fsync_dir"], os.replace

    def spy_file(p):
        calls["file"] += 1
        return of_file(p)

    def spy_dir(p):
        calls["dir"] += 1
        return of_dir(p)

    def spy_repl(a, b):
        calls["replace"] += 1
        return of_repl(a, b)

    g["_fsync_file"], g["_fsync_dir"], os.replace = spy_file, spy_dir, spy_repl
    try:
        with tempfile.TemporaryDirectory() as fd:
            p = os.path.join(fd, "d.jsonl")
            with open(p, "w") as fh:
                fh.write(json.dumps({"content": "durable please"}) + "\n")
            scan_shards(
                "code_py",
                [p],
                fd,
                const_pred,
                cfg,
                {
                    "docs": 0,
                    "dup_docs": 0,
                    "written_shards": 0,
                    "skipped_shards": 0,
                    "input_rows": 0,
                    "bad_content_rows": 0,
                },
            )
            # data file rename is one durable publish; marker rename is another -> >=1 each
            assert calls["file"] >= 1 and calls["dir"] >= 1 and calls["replace"] >= 1, calls
    finally:
        g["_fsync_file"], g["_fsync_dir"], os.replace = of_file, of_dir, of_repl

    print(
        "l2_census_scan selftest OK: disjoint sharding, idempotent fingerprint-checked "
        "resume (stale head/version/markerless refused without --force), canonical "
        "continuous rows unquantized (out-of-range kept), quota ranking separable, "
        "version-conflict guard, durable fsync publish, bad-row counting, empty/all-bad "
        "shard refusal with all-duplicate shards skipped, per-row + marker-persisted "
        "encoder over-ctx truncation flag, domain-level + cross-shard dup detection"
    )


if __name__ == "__main__":
    main()
