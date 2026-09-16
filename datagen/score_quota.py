#!/usr/bin/env python3
"""Per-domain/per-language CONDITIONAL quota selector over the frozen score ledger.

# restartable: a pure read over ledger rows -> one thresholds JSON written atomically at
# the end; an interrupt writes nothing and re-running recomputes from the ledger. It never
# scans a corpus shard or rewrites the ledger.

The funnel step AFTER scoring. A single global threshold wipes whole domains because
domains differ in score distributions; this module instead pins ONE scorer version and, for
each (domain, lang) group, picks its OWN threshold to meet that domain's target KEEP QUOTA.
It reads datagen/score_ledger.py and writes decisions; it holds no scorer math.

Ordering: scalar rows (score != null) order on score. Multi-dim rubric rows order on one
dim (--rubric-dim) or, when no dim is named, the mean of the dims. Exact quota conservation
comes from ranking on (value desc, doc_id asc) and taking the top k, so ties resolve
deterministically and exactly k survive (a value>=threshold cut over-keeps ties).

Guards are loud, never silent:
  EmptyDomain        a quota names a group the pinned scorer produced no scores for;
  DomainWouldEmpty   the quota keeps fewer than the group's survival floor;
  VersionConflict    one selection saw several scorer_versions (must pin one).

export_review_sample adds the feedback hook: 'high' audits false-high top scorers and
'boundary' lists docs straddling the cut (the hard-negative pool for the next scorer
iteration). It exports ids/scores only; no retraining here.

    python3 datagen/score_quota.py select ledger.jsonl --scorer kenlm --version v1 \
        --quota '{"en_c4_stage2_dc":0.6}' --out th.json
    python3 datagen/score_quota.py sample ledger.jsonl --thresholds th.json \
        --scorer kenlm --version v1 --out review.jsonl --mode boundary
    python3 datagen/score_quota.py --selftest
"""

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datagen.score_ledger import load_rows

MIN_KEEP_FRAC_DEFAULT = 0.01
MIN_KEEP_DOCS_DEFAULT = 1


class QuotaError(Exception):
    """Base class for loud selector refusals."""


class EmptyDomain(QuotaError):
    """A quota targets a (domain, lang) group with no pinned scores."""


class DomainWouldEmpty(QuotaError):
    """A quota/floor combination would keep fewer than the survival floor."""


class VersionConflict(QuotaError):
    """One selection saw several scorer_versions for the pinned scorer."""


def _order_value(row, rubric_dim):
    if row["score"] is not None:
        return float(row["score"])
    dims = row["rubric_dims"]
    if rubric_dim is not None:
        if rubric_dim not in dims:
            raise QuotaError(f"rubric_dim {rubric_dim!r} absent on {row['doc_id']}")
        return float(dims[rubric_dim])
    return sum(dims.values()) / len(dims)


def pin_groups(rows, scorer_name, scorer_version=None, rubric_kind=None, rubric_dim=None, by_lang=True):
    """Filter validated ledger rows to one scorer/version/kind, refuse an unpinned version
    mix, and return {(domain, lang): [(doc_id, value), ...]} plus the resolved version."""
    sel = [r for r in rows if r["scorer_name"] == scorer_name and r["rubric_kind"] == rubric_kind]
    versions = sorted({r["scorer_version"] for r in sel})
    if not sel:
        raise EmptyDomain(f"no ledger rows for scorer={scorer_name} rubric_kind={rubric_kind!r}")
    if scorer_version is not None:
        if scorer_version not in versions:
            raise VersionConflict(f"requested {scorer_version!r}, ledger has {versions}")
        sel = [r for r in sel if r["scorer_version"] == scorer_version]
    elif len(versions) > 1:
        raise VersionConflict(f"scorer {scorer_name} has versions {versions}; pin one")
    groups = {}
    # A document must appear at most once in the pinned scorer/version selection. An
    # append-only census ledger can carry a duplicate doc_id (re-run / double append); if it
    # did, keep/total would count that doc twice, retain it twice, and silently push another
    # document out -- the per-domain quota would not conserve. Loud, mirroring
    # l2_dataset.load_pairs: reconcile the source, never silently dedup a selection.
    seen = set()
    for r in sel:
        key = (r["domain"], r["lang"]) if by_lang else (r["domain"], None)
        identity = (key, r["doc_id"])
        if identity in seen:
            raise QuotaError(
                f"duplicate doc_id {r['doc_id']!r} in group {key} for "
                f"{scorer_name}/{r['scorer_version']}; reconcile the ledger (one score per doc "
                "per pinned scorer+version) before selecting a quota"
            )
        seen.add(identity)
        groups.setdefault(key, []).append((r["doc_id"], _order_value(r, rubric_dim)))
    return groups, (scorer_version or versions[0])


def _quota_for(quotas, domain, lang):
    if (domain, lang) in quotas:
        return quotas[(domain, lang)]
    if domain in quotas:
        return quotas[domain]
    return None


def select_thresholds(
    groups, quotas, min_keep_frac=MIN_KEEP_FRAC_DEFAULT, min_keep_docs=MIN_KEEP_DOCS_DEFAULT
):
    """Per-group threshold to meet each quota; returns {group: decision dict}."""
    decisions = {}
    for key, docs in sorted(groups.items()):
        domain, lang = key
        q = _quota_for(quotas, domain, lang)
        if q is None:
            continue
        if not docs:
            raise EmptyDomain(f"targeted group {key} has 0 scored documents")
        if not 0 < q <= 1:
            raise QuotaError(f"quota for {key} must be in (0,1], got {q}")
        total = len(docs)
        keep = max(1, min(total, math.ceil(q * total - 1e-9)))
        floor = max(min_keep_docs, math.ceil(min_keep_frac * total - 1e-9))
        if keep < floor:
            raise DomainWouldEmpty(
                f"group {key}: quota {q} keeps {keep}/{total} < floor {floor} "
                f"(min_frac={min_keep_frac}, min_docs={min_keep_docs})"
            )
        ranked = sorted(docs, key=lambda dv: (-dv[1], dv[0]))
        chosen = ranked[:keep]
        decisions[key] = {
            "domain": domain,
            "lang": lang,
            "quota": q,
            "total": total,
            "keep": keep,
            "keep_frac": round(keep / total, 6),
            "threshold": chosen[-1][1],
            "kept_doc_ids": [d for d, _ in chosen],
        }

    present_domains = {d for d, _ in groups}
    for qk in quotas:
        if isinstance(qk, tuple):
            if qk not in groups:
                raise EmptyDomain(f"quota names {qk} but the scorer scored none of it")
        elif qk not in present_domains:
            raise EmptyDomain(f"quota names domain {qk!r} but the scorer scored none of it")
    return decisions


def export_review_sample(groups, decisions, per_group, mode="high"):
    """doc_id/value/threshold checklist around each group's cut (no document content)."""
    if mode not in ("high", "boundary"):
        raise QuotaError(f"unknown mode {mode!r}; use high|boundary")
    out = []
    for key, dec in sorted(decisions.items()):
        kept_ids = set(dec["kept_doc_ids"])
        ranked = sorted(groups[key], key=lambda dv: (-dv[1], dv[0]))
        if mode == "high":
            # false-high audit: draw only from the KEPT set, capped at the kept count
            pick = ranked[: min(per_group, dec["keep"])]
        else:
            cut = dec["keep"]
            lo = max(0, cut - per_group // 2)
            pick = ranked[lo : cut + (per_group - (cut - lo))]
        for doc_id, value in pick:
            out.append(
                {
                    "domain": dec["domain"],
                    "lang": dec["lang"],
                    "doc_id": doc_id,
                    "score": value,
                    "threshold": dec["threshold"],
                    "kept": doc_id in kept_ids,
                    "mode": mode,
                }
            )
    return out


def parse_quota(raw):
    out = {}
    for k, v in json.loads(raw).items():
        if "|" in k:
            d, lang = k.split("|", 1)
            out[(d, lang)] = float(v)
        else:
            out[k] = float(v)
    return out


def _synth_rows():
    from datagen.score_ledger import ScoreRow

    def mk(doc_id, domain, lang, score, version="v1"):
        return ScoreRow(
            doc_id=doc_id,
            domain=domain,
            lang=lang,
            scorer_name="kenlm",
            scorer_version=version,
            ts="2026-09-16T00:00:00Z",
            score=score,
            stratum=None,
        ).to_dict()

    rows = []
    for i, s in enumerate([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]):
        rows.append(mk(f"we{i}", "web", "en", s))
    for i, s in enumerate([0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12]):
        rows.append(mk(f"wz{i}", "web", "zh", s))
    for i, s in enumerate([0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37]):
        rows.append(mk(f"co{i}", "code", "en", s))
    return rows


def _selftest():
    rows = _synth_rows()
    groups, ver = pin_groups(rows, "kenlm", "v1")
    assert ver == "v1"

    # quota conservation incl. ties resolved deterministically by doc_id
    tied = {("t", "en"): [(d, 0.5) for d in ("a", "b", "c", "d")]}
    d = select_thresholds(tied, {"t": 0.5})[("t", "en")]
    assert d["keep"] == 2 and len(set(d["kept_doc_ids"])) == 2

    decs = select_thresholds(groups, {"web": 0.5, "code": 0.25})
    assert decs[("web", "en")]["keep"] == 5
    assert decs[("web", "zh")]["keep"] == 4
    assert decs[("code", "en")]["keep"] == 2
    # conditional, not global: low-scored zh/web gets its own lower threshold
    assert decs[("web", "zh")]["threshold"] < decs[("web", "en")]["threshold"]

    for fn, kw in (
        (
            lambda: select_thresholds(
                {("web", "en"): [(f"w{i}", 0.1 * i) for i in range(1, 11)]}, {"web": 0.5, "ghost": 0.5}
            ),
            {},
        ),
        (
            lambda: select_thresholds(
                {("d", "en"): [(f"x{i}", 0.1) for i in range(100)]}, {"d": 0.001}, min_keep_frac=0.05
            ),
            {},
        ),
        (lambda: pin_groups(rows + [{**rows[0], "scorer_version": "v2"}], "kenlm"), {}),
        # duplicate doc_id in one pinned group must raise (quota would double-retain it)
        (lambda: pin_groups(rows + [dict(rows[0])], "kenlm", "v1"), {}),
    ):
        try:
            fn(**kw)
        except (EmptyDomain, DomainWouldEmpty, VersionConflict, QuotaError):
            pass
        else:
            raise AssertionError("expected a loud selector refusal")

    # rubric multi-dim ordering: select on one dim
    from datagen.score_ledger import ScoreRow

    rb = [
        ScoreRow(
            doc_id=f"r{i}",
            domain="py",
            lang="en",
            scorer_name="l3-rubric",
            scorer_version="r1",
            ts="2026-09-16T00:00:00Z",
            rubric_dims={"content_quality": i % 3 + 1, "complexity": 5 - (i % 5)},
            rubric_kind="py",
            stratum={"language": "en"},
        ).to_dict()
        for i in range(6)
    ]
    g2, _ = pin_groups(rb, "l3-rubric", "r1", rubric_kind="py", rubric_dim="complexity")
    d2 = select_thresholds(g2, {"py": 0.5})[("py", "en")]
    top_values = sorted((v for _, v in g2[("py", "en")]), reverse=True)[:3]
    assert d2["keep"] == 3 and d2["threshold"] == min(top_values)

    high = export_review_sample(groups, decs, 3, "high")
    assert len(high) == 8 and all(r["kept"] for r in high)
    boundary = export_review_sample(groups, decs, 4, "boundary")
    assert all(r["doc_id"] in dict(groups[(r["domain"], r["lang"])]) for r in boundary)

    print(
        "score_quota selftest OK: quota conservation, conditional per-lang thresholds, "
        "empty/wiped/version refusals, rubric-dim ordering, review export"
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    ps = sub.add_parser("select")
    ps.add_argument("ledger")
    ps.add_argument("--scorer", required=True)
    ps.add_argument("--version", default=None)
    ps.add_argument("--rubric-kind", default=None)
    ps.add_argument("--rubric-dim", default=None)
    ps.add_argument("--quota", required=True)
    ps.add_argument("--min-keep-frac", type=float, default=MIN_KEEP_FRAC_DEFAULT)
    ps.add_argument("--min-keep-docs", type=int, default=MIN_KEEP_DOCS_DEFAULT)
    ps.add_argument("--no-lang", action="store_true")
    ps.add_argument("--out", required=True)
    pm = sub.add_parser("sample")
    pm.add_argument("ledger")
    pm.add_argument("--thresholds", required=True)
    pm.add_argument("--scorer", required=True)
    pm.add_argument("--version", default=None)
    pm.add_argument("--rubric-kind", default=None)
    pm.add_argument("--rubric-dim", default=None)
    pm.add_argument("--no-lang", action="store_true")
    pm.add_argument("--per-group", type=int, default=20)
    pm.add_argument("--mode", choices=("high", "boundary"), default="high")
    pm.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.selftest:
        _selftest()
        return
    if a.cmd == "select":
        rows = load_rows(a.ledger)
        groups, ver = pin_groups(rows, a.scorer, a.version, a.rubric_kind, a.rubric_dim, not a.no_lang)
        decisions = select_thresholds(groups, parse_quota(a.quota), a.min_keep_frac, a.min_keep_docs)
        tmp = a.out + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "scorer_name": a.scorer,
                    "scorer_version": ver,
                    "rubric_kind": a.rubric_kind,
                    "rubric_dim": a.rubric_dim,
                    "decisions": list(decisions.values()),
                },
                f,
                ensure_ascii=False,
                indent=1,
            )
        os.replace(tmp, a.out)
        print(
            f"selected {len(decisions)} groups, keep "
            f"{sum(d['keep'] for d in decisions.values())} docs -> {a.out}"
        )
    elif a.cmd == "sample":
        rows = load_rows(a.ledger)
        groups, _ = pin_groups(rows, a.scorer, a.version, a.rubric_kind, a.rubric_dim, not a.no_lang)
        with open(a.thresholds, encoding="utf-8") as f:
            blob = json.load(f)
        decisions = {(d["domain"], d["lang"]): d for d in blob["decisions"]}
        sample = export_review_sample(groups, decisions, a.per_group, a.mode)
        with open(a.out, "w", encoding="utf-8") as f:
            for r in sample:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"exported {len(sample)} review rows ({a.mode}) -> {a.out}")
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
