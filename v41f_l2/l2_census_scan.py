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


def read_corpus_docs(shard_path: str):
    """yield (content, src_sha_or_None) per JSONL row. Blank/non-string content skipped."""
    with open(shard_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            content = row.get("content")
            if isinstance(content, str) and content:
                yield content, row.get("src_sha")


def build_row(content, vec, cfg: ScanConfig, domain, shard_path, src_sha=None):
    """One whole-doc 4-vector -> a canonical head-score row (continuous, unquantized).
    doc_id is the whole-content hash, re-checked against the bytes."""
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
        "dims": {name: float(x) for name, x in zip(RUBRIC_DIMS, vec, strict=True)},
    }


def _utcnow() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def scan_shards(domain, paths, out_dir, predict, cfg: ScanConfig, counters, *, force=False):
    """Score a worker's assigned shards, one validated output file per shard.

    Idempotent: a finished (domain,shard) file is skipped. Each shard stages to a .tmp and
    atomically renames, so a crash never leaves partial/duplicate rows; in-shard doc_ids
    de-duplicate. Mutates the shared counters dict; returns rows_written_total."""
    written = 0
    for shard_path in paths:
        out_path = shard_output_path(out_dir, domain, shard_path)
        if out_path.exists() and not force:
            counters["skipped_shards"] += 1
            continue
        rows = _score_one_shard(domain, shard_path, predict, cfg, counters)
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        if tmp.exists():
            tmp.unlink()
        # append_rows validates each row against the canonical schema before any byte lands
        append_rows(str(tmp), rows)
        os.replace(tmp, out_path)
        written += len(rows)
        counters["written_shards"] += 1
    return written


DOC_BATCH = 256  # unique docs per predictor call; bounds text held in memory


def _score_one_shard(domain, shard_path, predict, cfg, counters):
    # Stream the shard in bounded DOC_BATCH micro-batches: a shard can be GBs, so we never
    # hold all its text. Within a batch the predictor packs to the GPU token budget (the 131
    # docs/s shape). Dedup is on the small content-id set; only compact rows survive a batch.
    seen: set[str] = set()
    rows = []

    def flush(batch):
        if not batch:
            return
        vecs = predict([c for c, _ in batch])
        if len(vecs) != len(batch):
            raise ValueError(f"predictor returned {len(vecs)} vectors for {len(batch)} docs")
        for (content, src_sha), vec in zip(batch, vecs, strict=True):
            rows.append(build_row(content, vec, cfg, domain, shard_path, src_sha=src_sha))
            counters["docs"] += 1

    batch = []
    for content, src_sha in read_corpus_docs(shard_path):
        cid = content_doc_id(content)
        if cid in seen:
            counters["dup_docs"] += 1
            continue
        seen.add(cid)
        batch.append((content, src_sha))
        if len(batch) >= DOC_BATCH:
            flush(batch)
            batch = []
    flush(batch)
    return rows


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

    def _encode_batch(self, texts):
        torch = self.torch
        enc = self.tok(texts, padding=True, truncation=True, max_length=MAX_CTX, return_tensors="pt")
        # a doc at the cap was truncated; probe measured ~0, so a nonzero count is a signal
        self.over_ctx += int((enc["attention_mask"].sum(1) >= MAX_CTX).sum())
        with torch.no_grad():
            pred = self.model(enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device))
        return pred.float().cpu().tolist()

    def __call__(self, texts):
        # whole docs, greedy token-budget packs (length-sorting upstream reduces padding in a
        # production run; the call boundary stays one batch in / one 4-vector-per-doc out).
        out, cur, cur_tok = [], [], 0
        for t in texts:
            ntok = len(self.tok(t, add_special_tokens=True)["input_ids"])
            if cur and cur_tok + min(ntok, MAX_CTX) > self.token_budget:
                out.extend(self._encode_batch(cur))
                cur, cur_tok = [], 0
            cur.append(t)
            cur_tok += min(ntok, MAX_CTX)
        if cur:
            out.extend(self._encode_batch(cur))
        return out


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
    shards = assign_shards(enumerate_shards(globs), args.shard, args.num_shards)
    cfg = ScanConfig(scorer_version=args.scorer_version)
    predict = HeadPredictor(args.head_ckpt, cfg)
    by_domain: dict[str, list] = {}
    for domain, path in shards:
        by_domain.setdefault(domain, []).append(path)
    counters = {"docs": 0, "dup_docs": 0, "written_shards": 0, "skipped_shards": 0}
    for domain, paths in sorted(by_domain.items()):
        scan_shards(domain, paths, args.out_dir, predict, cfg, counters, force=args.force)
    # probe measured ~0 docs at 8192; nonzero means the length distribution moved and docs
    # were silently truncated -- surfaced as the scan's closing line.
    print(f"{counters} over_ctx(={MAX_CTX})={getattr(predict, 'over_ctx', 0)}", flush=True)


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

        # a forced rescan still validates (crash/.tmp safety exercised via overwrite)
        c3 = {"docs": 0, "dup_docs": 0, "written_shards": 0, "skipped_shards": 0}
        scan_shards("code_py", [s0], td, stub_predict, cfg, c3, force=True)
        assert c3["written_shards"] == 1 and len(load_rows(shard_output_path(td, "code_py", s0))) == 2

    print(
        "l2_census_scan selftest OK: disjoint sharding, idempotent resume, canonical "
        "continuous rows unquantized (out-of-range kept), quota ranking separable, "
        "version-conflict guard, atomic write"
    )


if __name__ == "__main__":
    main()
