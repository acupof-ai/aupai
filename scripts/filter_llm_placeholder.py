#!/usr/bin/env python3
"""Post-pass filter: drop LLM-rewriter placeholder / no-content meta-documents from
an already-cleaned corpus (swallow_math, 2026-10-05).

The tokyotech-llm/swallow-math rewriter (Llama-3.3-70B over FineMath-4+) emits, for
source pages with no extractable problem, a short META-DESCRIPTION saying there is no
specific math problem/question to solve ("There is no specific math problem to solve
in the provided text. The text appears to be a collection of ..."). These carry no
trainable math: they are the rewrite pipeline's refusal/empty result. They also
formed the largest MinHash near-dup cluster (597 docs in the 100k sample).

The rule is deliberately CONSERVATIVE (de filter discipline, 2026-10-05): match only
the small family of explicit "no math problem/question" declarations, anchored in the
document's first 250 chars, so genuine math prose that merely mentions a missing value
is kept. It does NOT use startswith on literal sentences -- the patterns are the
induced sentence SHAPES from ~30 hand-read examples. False-drop and missed-kept rates
are measured by two-sided 100-doc hand reads (scripts/swallow_math_*_100.jsonl).

Read-only by default; --apply rewrites the corpus in place (like near_dedup_postpass),
hardlinks nothing (content rewrite), writes a hits sample and a stamp with
template_dropped/scanned/fraction, and the rule source is included in the recorded
rule_fp so changing the rule changes the fingerprint.

# restartable: --apply streams each shard once and writes .part then renames; an
# interrupt leaves .part files that a re-run replaces, and shards not yet touched keep
# their bytes. The scan-only default holds no state.
"""

import argparse
import glob
import hashlib
import json
import os
import re

# The induced family. Every arm is an explicit declaration of ABSENCE OF A SOLVABLE
# MATH PROBLEM/QUESTION, in the rewriter's voice. Tested against the first HEAD chars
# only. Word-boundary tolerant; contractions included. Keep this list SMALL and
# semantic -- adding a loose arm is how the false-drop rate climbs.
HEAD = 250
_PATTERNS = [
    # "there is no specific math problem to solve (in the provided/given text)"
    r"there (?:is|are|was) no (?:specific |particular |actual |real |clear |single |actual )?"
    r"(?:math\w*|mathematical)? ?(?:problem|question|task|calculation|exercise)s? to (?:solve|answer|be solved|be answered)",
    # "the (provided/given) text does not contain a(ny) (specific) math problem"
    r"(?:the |this |that )?(?:provided|given|source|input|original)? ?text (?:does not|doesn't|did not) "
    r"(?:contain|include|present|have|state|provide) (?:a |an |any |specific |particular |clear |single |real )?"
    r"(?:math\w*|mathematical)? ?(?:problem|question|task|calculation|exercise)s?",
    # "does not contain a specific math problem / question" (no "text" subject variant)
    r"does not contain (?:a |any |specific |particular )?(?:math\w*|mathematical) "
    r"(?:problem|question|task|exercise)s?(?: to (?:solve|be solved))?",
    # "no specific math problem/question (is) provided/presented/asked/given"
    r"no (?:specific |particular |clear |actual )?(?:math\w*|mathematical) "
    r"(?:problem|question|task|exercise)s? (?:is |was |are |has been )?(?:provided|presented|asked|given|included|stated|specified|offered)",
    # "there is no specific question/problem presented/asked" (math omitted but in rewriter voice)
    r"there (?:is|was) no (?:specific |particular )?(?:question|problem|task) "
    r"(?:presented|asked|provided|given|specified|stated)",
    # "no specific math problem or question to be solved/answered" (conjoined object;
    # observed as a miss in the kept-100 audit)
    r"no (?:specific |particular )?(?:math\w*|mathematical)? ?"
    r"(?:problem|question|task)s? (?:or|/|and) (?:problem|question|task)s? "
    r"to (?:be )?(?:solve|answer|solved|answered)",
]
_RULE_RE = re.compile("|".join(f"(?:{p})" for p in _PATTERNS), re.IGNORECASE)


def rule_fp() -> str:
    """Content hash of the rule the filter applies (patterns + head window), so the
    stamp's decontam provenance changes when the rule changes."""
    h = hashlib.sha256()
    h.update(repr(HEAD).encode())
    for p in _PATTERNS:
        h.update(p.encode())
    return h.hexdigest()[:16]


def is_placeholder(text: str) -> bool:
    return bool(_RULE_RE.search(text[:HEAD]))


def _shard_paths(out: str, domain: str):
    return sorted(glob.glob(os.path.join(out, f"{domain}_*.jsonl")))


def scan(out: str, domain: str):
    """Return (scanned, dropped, per_shard {name: dropped}) without writing."""
    scanned = dropped = 0
    per = {}
    for p in _shard_paths(out, domain):
        n = d = 0
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                n += 1
                if is_placeholder(json.loads(line).get("content") or ""):
                    d += 1
        per[os.path.basename(p)] = d
        scanned += n
        dropped += d
    return scanned, dropped, per


def apply(out: str, domain: str, sample_keep: int = 100):
    """Rewrite every shard dropping placeholder rows; collect dropped samples and,
    from a disjoint stride, kept samples for the two-sided hand-read audit."""
    paths = _shard_paths(out, domain)
    if not paths:
        raise SystemExit(f"no {domain}_*.jsonl in {out}")
    scanned = dropped = 0
    dropped_samples, kept_samples = [], []
    for p in paths:
        tmp = p + ".part"
        n = d = 0
        with open(p, encoding="utf-8") as fin, open(tmp, "w", encoding="utf-8") as fout:
            for line in fin:
                s = line.strip()
                if not s:
                    continue
                n += 1
                row = json.loads(s)
                if is_placeholder(row.get("content") or ""):
                    d += 1
                    if len(dropped_samples) < 200:
                        dropped_samples.append(row.get("content", "")[:400])
                else:
                    fout.write(line if line.endswith("\n") else line + "\n")
                    # stride the kept sample so it is not just the first 100 rows
                    if n % 997 == 1 and len(kept_samples) < 400:
                        kept_samples.append(row.get("content", "")[:400])
        os.replace(tmp, p)
        scanned += n
        dropped += d
    _write_stamp(out, scanned, dropped, dropped_samples)
    return scanned, dropped


def _write_stamp(out, scanned, dropped, dropped_samples):
    """Write the template_filter stamp block and recompute the dir fingerprint.
    Shared by --apply and --stamp_only."""
    from corpus_fingerprint import fp_dir  # late import: run from repo root

    # audit artifacts live in a SIBLING dir: fp_dir hashes every non-dot top-level entry
    # by opening it as a file (a subdir crashes it), and the bare top-level *.jsonl
    # training/decontam glob would otherwise ingest the sample files.
    audit_dir = os.path.join(os.path.dirname(out.rstrip("/")), os.path.basename(out.rstrip("/")) + "_audit")
    os.makedirs(audit_dir, exist_ok=True)
    hits_path = os.path.join(audit_dir, "template_hits_sample.jsonl")
    if dropped_samples and not os.path.exists(hits_path):
        with open(hits_path, "w", encoding="utf-8") as f:
            for t in dropped_samples:
                f.write(json.dumps({"content": t}, ensure_ascii=False) + "\n")
    stamp = os.path.join(out, "build_corpus_stats.json")
    with open(stamp, encoding="utf-8") as sf:
        s = json.load(sf)
    s["template_filter"] = {
        "rule": "llm-no-content-meta-description",
        "rule_fp": rule_fp(),
        "head_chars": HEAD,
        "template_scanned": scanned,
        "template_dropped": dropped,
        "template_dropped_fraction": (dropped / scanned) if scanned else 0.0,
        "removed_fraction_note": (
            "template_dropped_fraction is the gross removed fraction "
            "and is an UPPER BOUND on real no-content docs: it includes "
            "the measured false-kill class below."
        ),
        # Two-sided hand-read audit (de filter discipline, 2026-10-05): 100 rule-hit +
        # 100 kept docs drawn by fixed strides and read in full. 2 of 100 hits were
        # false kills (mixed docs whose opening declares no problem but whose tail adds a
        # real worked problem); 0 of 100 kept were missed after the rule patch.
        "false_kill_audit": {
            "sample_hits_n": 100,
            "sample_kept_n": 100,
            "sampling": "fixed-stride samples across all shards; see scripts/swallow_math_placeholder_audit_sample.py",
            "false_kill_pct_of_hits": 2.0,
            "missed_in_kept_pct": 0.0,
            "false_kill_type": "opening declares no specific math problem/question, but the document tail adds a real worked problem/solution (mixed rewriter output)",
            "false_kill_examples_in_audit_file": [
                "ph_drop100 #53 (3D distance P/Q problem added in tail)",
                "ph drop100 #72 ('Here are the math problems with solutions', real word problems after one placeholder item)",
            ],
            "estimated_false_kills_full_domain": round(dropped * 0.02),
            "estimated_false_kill_fraction_of_domain": (dropped * 0.02 / scanned) if scanned else 0.0,
        },
        "hits_sample": "../swallow_math_audit/template_hits_sample.jsonl",
        "audit_samples": "../swallow_math_audit/template_audit_drop100.jsonl + ../swallow_math_audit/template_audit_keep100.jsonl (sibling dir; two-sided hand-read set; the two known false-kill rows #53/#72 retained verbatim)",
    }
    s["fingerprint"] = fp_dir(out)
    s["filters_fp"] = s.get("filters_fp", "")
    with open(stamp, "w", encoding="utf-8") as f:
        json.dump(s, f, ensure_ascii=False, indent=1)
    return scanned, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", help="corpus dir, e.g. data/corpus/swallow_math")
    ap.add_argument("--domain", default="swallow_math")
    ap.add_argument("--apply", action="store_true", help="rewrite shards (default: scan only)")
    ap.add_argument(
        "--stamp_only",
        action="store_true",
        help="do not rewrite; only write the stamp (shards already filtered). "
        "Requires --scanned/--dropped from the apply run.",
    )
    ap.add_argument("--scanned", type=int, default=0)
    ap.add_argument("--dropped", type=int, default=0)
    a = ap.parse_args()
    if a.stamp_only:
        if not (a.scanned and a.dropped):
            raise SystemExit("--stamp_only needs --scanned and --dropped from the apply run")
        # verify the surviving count matches scanned-dropped and no rule hit remains
        kept, hits_left, _ = scan(a.out, a.domain)
        if hits_left:
            raise SystemExit(
                f"{hits_left} rule hits still in shards; current rule differs "
                "from the applied rule_fp; refusing a mismatched stamp"
            )
        if kept != a.scanned - a.dropped:
            raise SystemExit(
                f"shards hold {kept} docs != scanned-dropped "
                f"{a.scanned - a.dropped}; refusing a stamp that does not match bytes"
            )
        _write_stamp(a.out, a.scanned, a.dropped, [])
        print(
            f"STAMP_ONLY scanned={a.scanned} dropped={a.dropped} kept={kept} rule_fp={rule_fp()} fp_updated",
            flush=True,
        )
    elif a.apply:
        scanned, dropped = apply(a.out, a.domain)
        print(
            f"APPLIED scanned={scanned} dropped={dropped} "
            f"fraction={dropped / scanned * 100:.4f}% rule_fp={rule_fp()} fp_updated",
            flush=True,
        )
    else:
        scanned, dropped, per = scan(a.out, a.domain)
        print(
            f"SCAN scanned={scanned} would_drop={dropped} "
            f"fraction={dropped / scanned * 100:.4f}% (max shard {max(per.values())})",
            flush=True,
        )
    print("DONE")


if __name__ == "__main__":
    main()
