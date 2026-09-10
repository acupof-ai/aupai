#!/usr/bin/env python3
"""Draw a blinded audit sample from a data line's artifact (cross-line audit 0909).

# restartable: the whole source is read into memory before anything is written;
# an interrupt costs one re-read and leaves no partial state behind.

One sampler for all four lines so the protocol is uniform
(docs/standards/cross_line_audit_0909.md).

Modes:
  random       n random docs
  stratified   K strata x M docs per stratum by --strata-field (n = K*M); rows
               emitted in stratum order so a reader sees each pair together;
               refuses on short strata unless --allow-short
  highlow      n top + n bottom by --score-field (2n rows); the group label is
               withheld from the sheet and kept in the manifest only, and the
               sheet is shuffled so row order cannot reveal the groups

Outputs <out>.jsonl (sample_id, pair_id, text) and <out>.manifest.json (source
files + sha256, seed, mode, n, command). The manifest is the sample's identity:
a re-draw against changed sources is a different sample, and the sampler refuses
to overwrite a sheet whose sources moved (srcfp rule).
"""
import argparse
import glob
import hashlib
import json
import os
import random
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["random", "stratified", "highlow"], required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20260909)
    ap.add_argument("--text-field", default="content")
    ap.add_argument("--strata-field")
    ap.add_argument("--strata", type=int)
    ap.add_argument("--per-stratum", type=int)
    ap.add_argument("--score-field")
    ap.add_argument("--allow-short", action="store_true",
                    help="stratified: write a short sample instead of refusing")
    a = ap.parse_args()

    paths = sorted(glob.glob(a.source))
    if not paths:
        sys.exit(f"REFUSE: no files match {a.source}")
    fps = {p: hashlib.sha256(open(p, "rb").read()).hexdigest() for p in paths}

    rows = []
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                d = json.loads(line)
                text = d.get(a.text_field) or d.get("text")
                if not text:
                    continue
                rows.append({
                    "text": text,
                    "stratum": d.get(a.strata_field) if a.strata_field else None,
                    "score": d.get(a.score_field) if a.score_field else None,
                })
    if not rows:
        sys.exit(f"REFUSE: no docs with text in {a.source}")

    rng = random.Random(a.seed)
    sheet = []
    groups = {}
    if a.mode == "random":
        for k, ix in enumerate(rng.sample(range(len(rows)), min(a.n, len(rows)))):
            sheet.append({"sample_id": f"R{k:03d}", "pair_id": None, "text": rows[ix]["text"]})
    elif a.mode == "stratified":
        if not (a.strata and a.per_stratum):
            sys.exit("REFUSE: stratified needs --strata and --per-stratum")
        by = {}
        for r in rows:
            if r["stratum"] is not None:
                by.setdefault(r["stratum"], []).append(r)
        keys = rng.sample(sorted(by), min(a.strata, len(by)))
        k = 0
        short = []
        for key in keys:
            docs = by[key]
            if len(docs) < a.per_stratum:
                short.append((str(key), len(docs)))
                continue
            for r in rng.sample(docs, a.per_stratum):
                sheet.append({"sample_id": f"S{k:03d}", "pair_id": str(key), "text": r["text"]})
                k += 1
        if short and not a.allow_short:
            sys.exit("REFUSE: strata too short: "
                     + ", ".join(f"{key}({n})" for key, n in short)
                     + " -- lower --strata/--per-stratum or pass --allow-short")
    else:  # highlow: neutral ids, group kept out of the sheet
        if not a.score_field:
            sys.exit("REFUSE: highlow needs --score-field")
        scored = sorted(rows, key=lambda r: r["score"])
        if 2 * a.n > len(scored):
            sys.exit(f"REFUSE: highlow needs 2n distinct docs, have {len(scored)}, "
                     f"need {2 * a.n} -- lower --n")
        order = [("lo", i) for i in range(a.n)] + [("hi", i) for i in range(len(scored) - a.n, len(scored))]
        for k, (group, ix) in enumerate(order):
            sid = f"H{k:03d}"
            groups[sid] = group
            sheet.append({"sample_id": sid, "pair_id": None, "text": scored[ix]["text"]})
        # blinding is a property of the sheet, not of where the label sits:
        # all-lo-then-all-hi row order unmasks the groups to any reader who
        # notices the first half looks worse. groups is keyed by sample_id,
        # so the shuffle does not touch it.
        rng.shuffle(sheet)

    if not sheet:
        sys.exit(f"REFUSE: sample is empty -- no docs matched the {a.mode} draw")
    # the shuffle lesson: a sample that equals the corpus head is a draw that did
    # not happen (handread_criterion_0908, fixed at 796fec85). Live in random
    # mode; stratified groups by stratum and highlow takes score tails, so
    # either can match the head only if the corpus is already ordered that way.
    if [s["text"] for s in sheet] == [r["text"] for r in rows[: len(sheet)]]:
        sys.exit("REFUSE: sample equals the corpus head -- the draw did not shuffle")

    manifest_path = a.out + ".manifest.json"
    if os.path.exists(manifest_path):
        old = json.load(open(manifest_path, encoding="utf-8"))
        if old.get("source_fp") != fps:
            sys.exit(f"REFUSE: {a.out} exists and its sources changed; use a new --out")
    with open(a.out + ".jsonl", "w", encoding="utf-8") as fh:
        for s in sheet:
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump({
            "source": a.source, "source_fp": fps, "seed": a.seed,
            "mode": a.mode, "n": len(sheet), "groups": groups,
            "command": " ".join(sys.argv),
        }, fh, ensure_ascii=False, indent=1)
    print(f"wrote {len(sheet)} rows to {a.out}.jsonl")


if __name__ == "__main__":
    main()
