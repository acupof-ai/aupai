#!/usr/bin/env python3
"""Counterfactual measurement: how many textbook chapters does the shipped
decontam normalise() miss because it strips markdown '#' headings and python
full-line comments before 13-gram shingling?

Read-only. Builds four normalisers and, for each, an independent benchmark
gram store, then scans every delivered chapter:

  N0 shipped          _COMMENT_LINE (^\\s*(?:#|//)) on every line, both sides
  N1 no strip at all  whitespace collapse only
  N2 fence-aware      strip full-line '#' comments ONLY inside a col-0 python
                      fence on chapters; strip '#' lines on benchmark python
                      source; keep markdown headings and all prose
  N3 keep everything  == N1 (asserted on the gram sets); used for attribution

Design: wf verify-decontam-normalise-gap 2026-09-13. Run on the pod.
  python3 scripts/decontam_md_counterfactual.py --out runs/decon_cf_0913.json

# restartable: read-only single pass over ~5k chapters writing one JSON at the
# end; an interrupt costs only a CPU-only rescan (~1 min), no shard state.
"""
import argparse
import glob
import hashlib
import json
import os
import re

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)

N = 13
_WS = re.compile(r"\s+")
HE = os.path.join(ROOT, "data", "eval", "humaneval", "humaneval_164.jsonl")
MBPP = os.path.join(ROOT, "data", "eval", "mbpp_holdouts.jsonl")
SRC = os.path.join(ROOT, "data", "corpus", "textbooks_claude_v41")

# N0 is the SHIPPED normaliser as of the measurement (pre de PR #324): full-line
# # and // comments deleted on both sides. decontam_ngram.normalise was changed
# to whitespace-only afterwards, so N0 is defined here, not imported, to keep the
# 0913 counterfactual reproducible.
_SHIPPED_COMMENT_LINE = re.compile(r"^\s*(?:#|//)\s?.*$", re.M)


def n0_normalise(s):
    return _WS.sub(" ", _SHIPPED_COMMENT_LINE.sub("", s or "")).strip()


_PY_HASH = re.compile(r"^\s*#\s?.*$")
_PY_LANGS = {"python", "py", "python3"}


def ws_only(s):
    return _WS.sub(" ", s or "").strip()


def py_strip_hash(s):
    """Benchmark python source under N2: drop full-line '#' comments, no fences."""
    kept = [ln for ln in (s or "").split("\n") if not _PY_HASH.match(ln)]
    return _WS.sub(" ", "\n".join(kept)).strip()


def fence_aware(s, any_indent=False, tilde=False, empty_is_py=True,
               strip_fence_comments=True):
    """Chapter markdown under N2/N3.

    Line scanner mirroring datagen/vet_textbooks.py FENCE pairing (non-nested,
    first opener to first closer). In a python fence, drop full-line '#'
    comments (N2); keep them for N3. Everything outside fences is prose/headings
    and kept byte-for-byte.
    """
    tick = "`~" if tilde else "`"
    out = []
    state = "OUT"
    for raw in (s or "").split("\n"):
        line = raw.rstrip("\r")
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        is_fence_line = False
        if tick == "`":
            opens = stripped.startswith("```")
        else:
            opens = stripped.startswith("```") or stripped.startswith("~~~")
        if state == "OUT":
            if opens and (indent == 0 or any_indent):
                fence_ch = line.strip()[0]
                if fence_ch == "~" and not tilde:
                    out.append(line)
                    continue
                info = stripped[3:].strip().lower()
                state = ("IN_PY" if (info in _PY_LANGS or (empty_is_py and info == ""))
                         else "IN_OTHER")
                is_fence_line = True
        elif state in ("IN_PY", "IN_OTHER") and (
            stripped.startswith("```") or (tilde and stripped.startswith("~~~"))
        ):
            state = "OUT"
            is_fence_line = True
        if is_fence_line:
            out.append(line)
        elif state == "IN_PY" and strip_fence_comments and _PY_HASH.match(line):
            continue
        else:
            out.append(line)
    return _WS.sub(" ", "\n".join(out)).strip()


def toks(norm_text):
    t = norm_text.strip()
    return _WS.split(t) if t else []


def grams(text, norm):
    t = toks(norm(text))
    if len(t) < N:
        return set()
    return {" ".join(t[i:i + N]) for i in range(len(t) - N + 1)}


# normaliser per (media, variant): variant k in N0..N3
PY_NORM = {"N0": n0_normalise, "N1": ws_only, "N2": py_strip_hash, "N3": ws_only}
PROSE_NORM = {"N0": n0_normalise, "N1": ws_only, "N2": ws_only, "N3": ws_only}


def md_norm(variant, **kw):
    if variant == "N0":
        return n0_normalise
    if variant == "N1" or variant == "N3":
        return ws_only
    return lambda s: fence_aware(s, **kw)


def _read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def _sha16(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()[:16]


def load_parts():
    """parts[variant][pid][part] = gram set, fields kept separate."""
    rows_he = _read_jsonl(HE)
    rows_mb = _read_jsonl(MBPP)
    parts = {v: {} for v in ("N0", "N1", "N2", "N3")}

    def add(v, pid, part, text, media):
        norm = PY_NORM[v] if media == "py" else PROSE_NORM[v]
        g = grams(text, norm)
        if g:
            parts[v].setdefault(pid, {})[part] = g

    for r in rows_he:
        pid = f"humaneval:{r['task_id']}"
        for v in parts:
            add(v, pid, "prompt", r.get("prompt", ""), "py")
            add(v, pid, "solution", r.get("canonical_solution", ""), "py")
            add(v, pid, "test", r.get("test", ""), "py")
    for r in rows_mb:
        pid = f"mbpp:{r['task_id']}"
        for v in parts:
            add(v, pid, "prompt", r.get("text", ""), "prose")
            add(v, pid, "solution", r.get("code", ""), "py")
    return parts


def scan(chapters, parts):
    """For each variant return {cid: {part: set(grams hit)}}."""
    hits = {v: {} for v in ("N0", "N1", "N2", "N3")}
    for cid, text in chapters:
        for v in ("N0", "N1", "N2", "N3"):
            cg = grams(text, md_norm(v))
            if not cg:
                continue
            found = {}
            for pid, pmap in parts[v].items():
                for part, bg in pmap.items():
                    inter = cg & bg
                    if inter:
                        found[f"{pid}|{part}"] = sorted(inter)
            if found:
                hits[v][cid] = found
    return hits


def scan_arms(chapters, parts_n2):
    """N2 sensitivity arms: only chapters that hit change; rebuild chapter grams
    per arm but reuse the single N2 benchmark store (benchmark side is identical
    across these markdown-parse variants)."""
    arms = {
        "primary": dict(),
        "any_indent": dict(any_indent=True),
        "tilde": dict(tilde=True),
        "strict_lang": dict(empty_is_py=False),
        "keep_fence_comments": dict(strip_fence_comments=False),
    }
    # flat benchmark gram set (part -> grams is enough for a hit count)
    bench = {}
    for pmap in parts_n2.values():
        for part, g in pmap.items():
            bench.setdefault(part, set()).update(g)
    bench_all = set().union(*bench.values()) if bench else set()
    res = {}
    for name, kw in arms.items():
        hit_ids = set()
        for cid, text in chapters:
            if grams(text, lambda s, k=kw: fence_aware(s, **k)) & bench_all:
                hit_ids.add(cid)
        res[name] = sorted(hit_ids)
    return res


def cid_of(source_file, idx, text):
    return f"{os.path.basename(source_file)}:{idx}:{hashlib.sha1(text.encode()).hexdigest()[:10]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    parts = load_parts()

    # identity: N3 benchmark grams must equal N1 for every part (both ws_only)
    assert parts["N3"] == parts["N1"], "N3 must equal N1 on benchmark store"

    chapters = []
    files = sorted(glob.glob(os.path.join(args.src, "gen*.jsonl")))
    per_file = {}
    for f in files:
        n = 0
        with open(f, encoding="utf-8") as fh:
            for idx, line in enumerate(fh):
                if not line.strip():
                    continue
                r = json.loads(line)
                cid = cid_of(f, idx, r.get("text", ""))
                chapters.append((cid, r.get("text", "")))
                n += 1
        per_file[os.path.basename(f)] = n

    hits = scan(chapters, parts)
    s = {v: set(hits[v]) for v in hits}

    # S0 must be empty on raw SOURCE? No -- source includes rows N0 drops; the
    # empty-S0 guarantee holds on the VETTED dir. Here record S0 for source.
    new1, new2 = s["N1"] - s["N0"], s["N2"] - s["N0"]
    H = (s["N2"] - s["N0"]) & s["N1"]            # caught under N2 and N1
    C = s["N1"] - s["N2"]                         # only with fence comments kept
    J = (s["N2"] - s["N0"]) - s["N1"]            # join-only artifact

    def per_problem(ids):
        cnt = {}
        parts_hit = {}
        for cid in ids:
            for key in hits["N1"].get(cid, {}):
                pid, part = key.split("|", 1)
                cnt[pid] = cnt.get(pid, 0) + 1
                parts_hit[part] = parts_hit.get(part, 0) + 1
        return cnt, parts_hit

    prob1, partp1 = per_problem(new1)

    arms = scan_arms(chapters, parts["N2"])

    result = {
        "universe": "delivered SOURCE chapters (pre-vet)",
        "n_chapters": len(chapters),
        "files": per_file,
        "n_gram": N,
        "benchmark_sha": {
            "humaneval": _sha16(HE),
            "mbpp": _sha16(MBPP),
            "decontam_ngram_py": _sha16(
                os.path.join(ROOT, "filters", "decontam_ngram.py")
            ),
        },
        "hits": {"N0_shipped": len(s["N0"]), "N1_no_strip": len(s["N1"]),
                 "N2_fence_aware": len(s["N2"]), "N3_keep_all": len(s["N3"])},
        "newly_caught": {
            "N1_minus_N0": len(new1),
            "N2_minus_N0": len(new2),
            "H_heading_or_prose": len(H),
            "C_comment_driven": len(C),
            "J_join_only_artifact": len(J),
        },
        "distinct_benchmark_problems_N1": len(prob1),
        "per_part_N1": partp1,
        "N2_arms_hit_counts": {k: len(v) for k, v in arms.items()},
        "N2_arm_vs_primary": {
            k: len(set(arms[k]) ^ set(arms["primary"])) for k in arms if k != "primary"
        },
        "new_chapters_N1_detail": {
            cid: {k: v[:3] for k, v in list(hits["N1"].get(cid, {}).items())[:4]}
            for cid in sorted(new1)
        },
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)
    print(json.dumps({k: result[k] for k in
                      ("n_chapters", "hits", "newly_caught",
                       "distinct_benchmark_problems_N1", "N2_arms_hit_counts",
                       "N2_arm_vs_primary")}, indent=2))


if __name__ == "__main__":
    main()
