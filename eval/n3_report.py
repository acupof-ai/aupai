#!/usr/bin/env python3
# restartable: reads summary JSONs and text files, writes one report file. Seconds, no GPU, no model.
"""N3's one-line-per-checkpoint report for the three benchmark-v2 metrics.

    python3 eval/n3_report.py --metric humaneval_bpb --summaries a.json b.json
    python3 eval/n3_report.py --metric math_bpb --summaries s7000.json s10000.json
    python3 eval/n3_report.py --selftest

FORMAT, fixed by roadmap_0903.md N3: per checkpoint one line carrying bits/byte, bytes saved
against the PREVIOUS point, and gzip -9 plus Pythia-160M on the same bytes beside it. The shape
is humaneval's, already set in roadmap §1 (0.5451 -> 0.5199 over 2021 -> 1928 bytes for 164
solutions, anchors gzip -9 2.096 / bzip2 1.897 / Pythia-160M 0.918 on the same 29,662 bytes), so
math_bpb reports in the same shape rather than inventing one.

WHY THE ANCHORS ARE COMPUTED HERE AND NOT QUOTED. Those three anchor numbers exist in exactly
one place in this repo -- a prose row in roadmap_0903.md -- with no script that produces them.
A number nobody can recompute cannot be checked when the corpus moves, and a byte total that
changed silently would leave the anchor comparing against a different string. gzip is computed
from the same concatenation the metric scores; bzip2 too, because it was reported alongside.

The PYTHIA anchor is NOT computed here: it needs a card and a model, so it is read from a
summary JSON produced by the same metric script with --hf. Passing it as a number would let a
figure measured on other bytes sit in this table -- so this script takes the control's summary
file and REFUSES if its byte total does not match our arm's.

BYTES SAVED is the honest unit for "did the model improve": bits/byte times bytes, differenced
against the previous checkpoint and divided by 8. A BPB delta of 0.03 sounds small and is 1.4 KB
on this corpus; both are printed because neither alone tells you the size of the effect.
"""

import argparse
import bz2
import gzip
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# metric name -> (summary key for bits/byte, summary key for the byte total, corpus loader)
METRICS = {
    "humaneval_bpb": ("gold_bpb_byte_weighted", "total_solution_bytes", "humaneval"),
    "math_bpb": ("math_bpb_byte_weighted", "total_gold_bytes", "math_test_500"),
}


def corpus_bytes(which):
    """The exact bytes the metric scores, concatenated, for the compression anchors.

    Concatenated in FILE ORDER and with no separator, matching how the metric sums per-item
    bits: an anchor computed on a different string is not an anchor for this number. Returns
    None when the source file is absent -- the anchors are then omitted and the report says so,
    rather than printing a compression ratio for bytes nobody has.
    """
    if which == "humaneval":
        p = os.path.join(ROOT, "data", "eval", "humaneval", "humaneval_164.jsonl")
        field = "canonical_solution"
        skip = 0
    else:
        p = os.path.join(ROOT, "data", "eval", "math_test_500.jsonl")
        field = "output"
        skip = 3   # math_bpb excludes l1_fewshot's three demos; the anchor scores what it scores
    if not os.path.exists(p):
        return None
    rows = [json.loads(x) for x in open(p, encoding="utf-8") if x.strip()]
    return "".join(r[field] for r in rows[skip:]).encode("utf-8")


def anchors(raw):
    """gzip -9 and bzip2 bits per byte on `raw`. None when there are no bytes to compress."""
    if not raw:
        return {}
    return {"gzip9_bpb": len(gzip.compress(raw, 9)) * 8 / len(raw),
            "bzip2_bpb": len(bz2.compress(raw)) * 8 / len(raw)}


def rows_from(summaries, bpb_key, bytes_key):
    out = []
    for path in summaries:
        d = json.load(open(path, encoding="utf-8"))
        if bpb_key not in d:
            sys.exit(f"REFUSING: {path} has no {bpb_key!r}. Keys: {sorted(d)[:8]}. A summary from "
                     f"a different metric would be reported under this metric's name.")
        out.append({"ckpt": d.get("ckpt", path), "bpb": d[bpb_key],
                    "bytes": d.get(bytes_key), "hf": d.get("hf", False), "summary": path})
    return out


def report(metric, summaries, control_summary=None, raw=None):
    """`raw` overrides the scored corpus; the selftest passes its own so its synthetic byte
    totals are not compared against the real 379,651-byte file. Production leaves it None."""
    bpb_key, bytes_key, which = METRICS[metric]
    pts = rows_from(summaries, bpb_key, bytes_key)
    if raw is None:
        raw = corpus_bytes(which)
    a = anchors(raw)

    # ONE BYTE TOTAL ACROSS EVERY POINT, or the deltas are differences of different corpora.
    totals = {p["bytes"] for p in pts if p["bytes"] is not None}
    if len(totals) > 1:
        sys.exit(f"REFUSING: the summaries disagree on {bytes_key}: {sorted(totals)}. Bytes saved "
                 f"between two points is meaningless when the points scored different corpora.")
    n_bytes = totals.pop() if totals else (len(raw) if raw else None)
    if raw and n_bytes and len(raw) != n_bytes:
        sys.exit(f"REFUSING: the anchor corpus is {len(raw)} bytes and the summaries report "
                 f"{n_bytes}. The anchors would describe a different string than the metric.")

    ctrl = None
    if control_summary:
        c = rows_from([control_summary], bpb_key, bytes_key)[0]
        if not c["hf"]:
            sys.exit(f"REFUSING: {control_summary} has hf=false, so it is not a control-arm run.")
        if n_bytes and c["bytes"] != n_bytes:
            sys.exit(f"REFUSING: control scored {c['bytes']} bytes, our arm {n_bytes}. A "
                     f"cross-tokenizer comparison is only honest on the same bytes.")
        ctrl = c

    lines = []
    for i, p in enumerate(pts):
        saved = ""
        if i:
            d_bits = (pts[i - 1]["bpb"] - p["bpb"]) * (n_bytes or 0)
            saved = f" | {d_bits / 8:+,.0f} B vs prev"
        extra = "".join(f" | {k.replace('_bpb', '')} {v:.3f}" for k, v in a.items())
        if ctrl:
            extra += f" | pythia160m {ctrl['bpb']:.3f}"
        lines.append(f"{os.path.basename(str(p['ckpt'])):<44} {p['bpb']:.4f} bits/B{saved}{extra}")
    return {"metric": metric, "n_bytes": n_bytes, "anchors": a,
            "control": ctrl["bpb"] if ctrl else None,
            "points": [{"ckpt": p["ckpt"], "bpb": p["bpb"], "summary": p["summary"]} for p in pts],
            "lines": lines}


def _selftest():
    import tempfile

    def summary(path, bpb, nbytes, hf=False, key="math_bpb_byte_weighted",
                bkey="total_gold_bytes", ckpt="c"):
        json.dump({key: bpb, bkey: nbytes, "hf": hf, "ckpt": ckpt},
                  open(path, "w", encoding="utf-8"))

    d = tempfile.mkdtemp()
    a_, b_ = os.path.join(d, "a.json"), os.path.join(d, "b.json")
    summary(a_, 1.0, 1000, ckpt="step1")
    summary(b_, 0.992, 1000, ckpt="step2")

    # BYTES SAVED = delta_bpb * bytes / 8. 0.008 bits/byte over 1000 bytes is exactly 1 byte.
    FIX = b"x" * 1000   # the fixture's own corpus, matching its summaries' byte total
    r = report("math_bpb", [a_, b_], raw=FIX)
    assert "+1 B vs prev" in r["lines"][1], r["lines"][1]
    # The FIRST point has no previous point, so it must not carry a saving.
    assert "vs prev" not in r["lines"][0], r["lines"][0]

    # A WORSE checkpoint reports a NEGATIVE saving rather than an absolute value: the sign is the
    # direction of the effect, and printing |delta| would make a regression read as an improvement.
    summary(b_, 1.008, 1000, ckpt="step2")
    r = report("math_bpb", [a_, b_], raw=FIX)
    assert "-1 B vs prev" in r["lines"][1], r["lines"][1]

    # DISAGREEING BYTE TOTALS REFUSE. Two points scoring different corpora cannot be differenced.
    summary(b_, 0.9, 999, ckpt="step2")
    try:
        report("math_bpb", [a_, b_], raw=FIX)
        raise AssertionError("two different byte totals were accepted")
    except SystemExit as e:
        assert "disagree" in str(e), e

    # A SUMMARY FROM THE WRONG METRIC REFUSES rather than being reported under this name.
    summary(b_, 0.9, 1000, key="gold_bpb_byte_weighted", bkey="total_solution_bytes")
    try:
        report("math_bpb", [a_, b_], raw=FIX)
        raise AssertionError("a humaneval summary was accepted as math_bpb")
    except SystemExit as e:
        assert "has no" in str(e), e

    # THE CONTROL MUST BE AN --hf RUN ON THE SAME BYTES.
    summary(b_, 0.992, 1000, ckpt="step2")
    ctrl = os.path.join(d, "c.json")
    summary(ctrl, 0.9, 1000, hf=False)
    try:
        report("math_bpb", [a_], control_summary=ctrl, raw=FIX)
        raise AssertionError("a non-hf summary was accepted as the control arm")
    except SystemExit as e:
        assert "hf=false" in str(e), e
    summary(ctrl, 0.9, 900, hf=True)
    try:
        report("math_bpb", [a_], control_summary=ctrl, raw=FIX)
        raise AssertionError("a control on different bytes was accepted")
    except SystemExit as e:
        assert "same bytes" in str(e), e
    summary(ctrl, 0.9, 1000, hf=True)
    r = report("math_bpb", [a_], control_summary=ctrl, raw=FIX)
    assert "pythia160m 0.900" in r["lines"][0], r["lines"][0]

    # THE ANCHORS ARE COMPUTED, and a compressor's bits/byte is bounded by 8 on any real input.
    raw = b"def f():\n    return 1\n" * 40
    got = anchors(raw)
    assert 0 < got["gzip9_bpb"] < 8, got
    assert 0 < got["bzip2_bpb"] < 8, got
    # Highly repetitive input must compress well below the 8 bits/byte of the raw encoding --
    # an anchor that read ~8 would mean the compressor never ran.
    assert got["gzip9_bpb"] < 1.0, got
    assert anchors(b"") == {}, "empty input must yield no anchors rather than a division by zero"

    for f in (a_, b_, ctrl):
        os.unlink(f)
    os.rmdir(d)
    print("n3_report self-test OK: bytes saved is delta_bpb x bytes / 8 with its sign kept, the "
          "first point carries none, disagreeing byte totals and wrong-metric summaries refuse, "
          "the control must be an --hf run on the same bytes, and the compression anchors are "
          "computed rather than quoted")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", choices=sorted(METRICS))
    ap.add_argument("--summaries", nargs="+", help="metric summary JSONs, in checkpoint order")
    ap.add_argument("--control", help="the same metric's summary from an --hf run (Pythia-160M)")
    ap.add_argument("--out", help="write the report JSON here")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
        return 0
    if not a.metric or not a.summaries:
        ap.error("--metric and --summaries required (or --selftest)")

    r = report(a.metric, a.summaries, a.control)
    print(f"{a.metric}: {r['n_bytes']:,} bytes" if r["n_bytes"] else f"{a.metric}")
    for line in r["lines"]:
        print(line)
    if not r["anchors"]:
        print("(no compression anchors: the corpus file is absent, so they were not computed "
              "rather than quoted from prose)")
    if r["control"] is None:
        print("(no Pythia-160M column: pass --control with an --hf summary of the same metric)")
    if a.out:
        json.dump(r, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
