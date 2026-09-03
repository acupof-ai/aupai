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


# The two preds formats name their item key differently; both are read here rather than one being
# renamed, because renaming would orphan every preds file already written.
ITEM_KEYS = ("id", "task_id")


def paired_se(preds_a, preds_b):
    """Byte-weighted BPB delta between two checkpoints, and the PAIRED SE of that delta.

    N3's sigma after 6e's ruling of 2026-09-03: the user cut N7 to inference-only, so no
    same-recipe seed pair exists anywhere and seed sigma cannot be measured. This is the other
    sigma -- the one that says whether THIS delta is bigger than its own item-to-item noise -- and
    it is NOT a substitute. Seed sigma answers "would a rerun with another seed move this much";
    this answers "do these items agree on the direction". A report must label the first unmeasured
    rather than print the second in its place.

    PAIRED, on the intersection of item ids, because the two checkpoints score the same items: the
    unpaired SE of two means would carry the between-item variance that cancels here, and on a
    corpus whose golds run from tens to thousands of bytes that variance dwarfs the effect.
    Byte-weighted to match the reported figure -- an SE for the per-item mean would not be the SE
    of the number beside it.
    """
    def load(path):
        out = {}
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            key = next((r[k] for k in ITEM_KEYS if k in r), None)
            if key is None or r.get("bpb") is None:
                continue
            out[key] = (float(r["bpb"]), int(r["n_bytes"]))
        return out

    A, B = load(preds_a), load(preds_b)
    shared = sorted(set(A) & set(B))
    if not shared:
        sys.exit(f"REFUSING: {preds_a} and {preds_b} share no item ids, so there is no paired "
                 f"comparison to make.")
    # BYTES FROM ONE SIDE, asserted equal: the same item must have the same gold bytes in both
    # runs. A mismatch means the two runs scored different text under one id, and the delta would
    # be a difference of two corpora wearing matching names.
    bad = [k for k in shared if A[k][1] != B[k][1]]
    if bad:
        sys.exit(f"REFUSING: {len(bad)} item(s) have different n_bytes in the two runs "
                 f"(e.g. {bad[0]}: {A[bad[0]][1]} vs {B[bad[0]][1]}). Same id, different text.")
    w = [A[k][1] for k in shared]
    tot = sum(w)
    # Per-item contribution to the byte-weighted delta, in bits: (bpb_a - bpb_b) * bytes. The
    # weighted delta is their sum over total bytes; the SE of that same expression is the sample
    # SD of the per-item contributions scaled by sqrt(n)/tot.
    contrib = [(A[k][0] - B[k][0]) * A[k][1] for k in shared]
    n = len(contrib)
    delta = sum(contrib) / tot
    if n < 2:
        return {"n_items": n, "delta_bpb": delta, "paired_se": None,
                "note": "one shared item: an SE needs at least two"}
    mean = sum(contrib) / n
    var = sum((c - mean) ** 2 for c in contrib) / (n - 1)
    se = (var * n) ** 0.5 / tot
    return {"n_items": n, "total_bytes": tot, "delta_bpb": delta, "paired_se": se,
            "z": delta / se if se else None,
            "bytes_saved": sum(contrib) / 8,
            "seed_sigma": "UNMEASURED -- no same-recipe seed pair exists (N7 was cut to "
                          "inference-only, 6e 2026-09-03). This SE is item-to-item noise on the "
                          "same items, NOT seed noise, and does not stand in for it."}


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

    # PAIRED SE. Two preds files over the same items: an identical pair must give delta 0, and a
    # constant shift must give delta = shift with SE 0, because every item moved the same amount.
    pa, pb = os.path.join(d, "pa.jsonl"), os.path.join(d, "pb.jsonl")

    def preds(path, vals, key="id"):
        with open(path, "w", encoding="utf-8") as f:
            for i, (v, nb) in enumerate(vals):
                f.write(json.dumps({key: f"x/{i}", "bpb": v, "n_bytes": nb}) + "\n")

    preds(pa, [(1.0, 10), (2.0, 90)])
    preds(pb, [(1.0, 10), (2.0, 90)])
    q = paired_se(pa, pb)
    assert abs(q["delta_bpb"]) < 1e-12 and abs(q["paired_se"]) < 1e-12, q

    preds(pb, [(0.9, 10), (1.9, 90)])
    q = paired_se(pa, pb)
    assert abs(q["delta_bpb"] - 0.1) < 1e-12, q
    # THE SE IS NOT ZERO HERE, and my first assertion said it should be. A constant BPB shift is
    # not a constant CONTRIBUTION: contributions are delta_bpb * bytes, so the 10-byte and 90-byte
    # items contribute 1.0 and 9.0 bits and legitimately differ. Zero SE requires equal
    # contributions, which means equal bytes -- checked separately below.
    assert q["paired_se"] > 0, q
    preds(pa, [(1.0, 50), (2.0, 50)])
    preds(pb, [(0.9, 50), (1.9, 50)])
    q = paired_se(pa, pb)
    assert abs(q["delta_bpb"] - 0.1) < 1e-12, q
    assert abs(q["paired_se"]) < 1e-12, f"equal bytes + equal shift must give zero SE, got {q}"
    preds(pa, [(1.0, 10), (2.0, 90)])   # restore the uneven fixture for the tests below
    # BYTE-WEIGHTED, not per-item: moving only the 90-byte item by 0.1 must give 0.09, which the
    # per-item mean would report as 0.05.
    # THREE UNEVEN ITEMS, so `sum/tot` and any `sum/n/const` disagree. A two-item fixture whose
    # bytes total 100 lets `sum(contrib)/n/50` equal `sum(contrib)/tot` exactly -- that mutation
    # stayed GREEN, which proved the fixture's arithmetic rather than the weighting.
    preds(pa, [(1.0, 10), (2.0, 90), (3.0, 400)])
    preds(pb, [(1.0, 10), (1.9, 90), (3.0, 400)])
    q = paired_se(pa, pb)
    want = 0.1 * 90 / 500
    assert abs(q["delta_bpb"] - want) < 1e-12, f"byte-weighted delta should be {want}, got {q}"
    assert abs(q["delta_bpb"] - 0.1 / 3) > 1e-6, "this is the per-item mean, not byte-weighted"
    preds(pa, [(1.0, 10), (2.0, 90)])
    preds(pb, [(1.0, 10), (1.9, 90)])
    q = paired_se(pa, pb)
    assert abs(q["delta_bpb"] - 0.09) < 1e-12, q
    assert q["paired_se"] > 0, "items disagreeing must give a non-zero SE"
    # SEED SIGMA IS LABELLED UNMEASURED and never filled with this number.
    assert "UNMEASURED" in q["seed_sigma"], q["seed_sigma"]
    # SAME ID, DIFFERENT BYTES REFUSES: that is two corpora under matching names.
    preds(pb, [(1.0, 10), (1.9, 91)])
    try:
        paired_se(pa, pb)
        raise AssertionError("mismatched n_bytes was accepted")
    except SystemExit as e:
        assert "different n_bytes" in str(e), e
    # NO SHARED IDS REFUSES rather than reporting a delta over nothing.
    preds(pb, [(1.0, 10)], key="task_id")
    q = paired_se(pa, pb)   # task_id is a recognised key, so x/0 still pairs
    assert q["n_items"] == 1 and q["paired_se"] is None, q
    for f in (pa, pb):
        os.unlink(f)

    for f in (a_, b_, ctrl):
        os.unlink(f)
    os.rmdir(d)
    print("n3_report self-test OK: bytes saved is delta_bpb x bytes / 8 with its sign kept, the "
          "first point carries none, disagreeing byte totals and wrong-metric summaries refuse, "
          "the control must be an --hf run on the same bytes, and the compression anchors are "
          "computed rather than quoted; and the paired SE is byte-weighted, zero only when equal "
          "shifts land on equal bytes (a constant BPB shift over UNEQUAL bytes gives a non-zero "
          "SE, which corrected my first assertion), refuses on same-id-different-bytes, and "
          "labels seed sigma unmeasured rather than substituting itself")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", choices=sorted(METRICS))
    ap.add_argument("--summaries", nargs="+", help="metric summary JSONs, in checkpoint order")
    ap.add_argument("--control", help="the same metric's summary from an --hf run (Pythia-160M)")
    ap.add_argument("--paired", nargs=2, metavar=("PREDS_A", "PREDS_B"),
                    help="two per-item preds files from the SAME metric: reports the byte-weighted "
                         "delta with its PAIRED SE. Seed sigma is unmeasurable (no same-recipe "
                         "seed pair exists) and is labelled unmeasured, never replaced by this")
    ap.add_argument("--out", help="write the report JSON here")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
        return 0
    if not a.metric or not a.summaries:
        ap.error("--metric and --summaries required (or --selftest)")

    r = report(a.metric, a.summaries, a.control)
    if a.paired:
        r["paired"] = paired_se(*a.paired)
    print(f"{a.metric}: {r['n_bytes']:,} bytes" if r["n_bytes"] else f"{a.metric}")
    for line in r["lines"]:
        print(line)
    if not r["anchors"]:
        print("(no compression anchors: the corpus file is absent, so they were not computed "
              "rather than quoted from prose)")
    if r.get("paired"):
        q = r["paired"]
        print(f"paired delta {q['delta_bpb']:+.4f} bits/B over {q['n_items']} shared items"
              + (f", SE {q['paired_se']:.4f}, z {q['z']:+.2f}" if q.get("paired_se") else "")
              + (f", {q['bytes_saved']:+,.0f} B" if q.get("bytes_saved") is not None else ""))
        print(f"seed sigma: {q['seed_sigma']}" if "seed_sigma" in q else "")
    if r["control"] is None:
        print("(no Pythia-160M column: pass --control with an --hf summary of the same metric)")
    if a.out:
        json.dump(r, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
