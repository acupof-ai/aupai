#!/usr/bin/env python3
"""e1-28: did our pretraining corpus already contain the held-out completions?

# restartable: builds an in-memory n-gram set over ~4.9 GB of text and prints counts; writes one
# small JSON. An interrupt costs the scan (10-15 min, CPU only) and nothing else -- no checkpoint,
# no GPU, no partial file that a later run could mistake for complete.

WHY THIS EXISTS. Section 4 of docs/audits/control_pythia160m_vs_ours.md registers a confound it
does not resolve: the leak check covered train<->heldout and two benchmarks, NOT either arm's
pretraining corpus. That matters most for section 5.0's two floors, because the floors compare two
UN-SFT'd models on this held-out text, and the 2.00x floor gap is the entire basis for "the gap is
mostly made in pretraining". If our side overlaps, our floor is optimistic and that conclusion
weakens.

THE POPULATION IS THE ROWS THE MODEL ACTUALLY READ, not the corpus on disk.
ckpt_p200m_4b_0902.pt carries row_cursor: 1,189,548 rows across nine domains, with a per-domain
row_cursor_srcfp. data/corpus is 232 GB while the run consumed 4B tokens, so scanning all of it
would report overlap from rows the model never saw -- inflating the answer in the direction that
makes our floor look worse-explained. Having the cursor means not choosing between "cannot finish"
and "measures the wrong thing".

srcfp IS CHECKED FIRST. If a domain's file changed since that run, the cursor's row numbers no
longer name the rows that were consumed, and a count over them is a count over something else.
Such domains are reported separately, never folded into the total.

TWO UNITS, BECAUSE ONE UNIT IS TWO DIFFERENT TESTS (1e's ruling, and the measurement behind it).
A 13-gram of whitespace tokens covers, per domain (200-doc probe of each first shard):

    zh_web 754 chars | chatml 195 | code_py_starcoder 93 | en_c4 77 | math_owm 74 | cot 70

zh_web averages 48.1 chars per whitespace token against English's 5.9, so its 13-gram is a ~10x
stricter test: a zero there would mean "the unit was wrong", not "no leak". So zh_web, chatml and
chat_qa additionally get a 13-CHARACTER sliding window, the two unit's hits are reported
separately, and either one counts as a hit. Code is left on the whitespace unit deliberately --
its 13 tokens are syntactic units, and a character window would shred indentation and identifiers
into matches that mean nothing.
"""
import argparse
import glob
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHAR_WINDOW_DOMAINS = {"zh_web", "chatml", "chat_qa"}
NGRAM = 13


def srcfp(path):
    """The same shape train.py's _corpus_fp uses: content, not mtime."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def domain_fp(d, root=None):
    files = sorted(glob.glob(os.path.join(root or ROOT, "data", "corpus", d, "*.jsonl")))
    h = hashlib.sha256()
    for p in files:
        h.update(srcfp(p).encode())
    return h.hexdigest()[:16], files


def text_of(rec):
    """The field is `content`. Reading `text` returned "" for every record and printed
    "no docs" for all six domains -- a missing key and an empty corpus look identical
    downstream, so this refuses instead of counting zero."""
    t = rec.get("content") or rec.get("text")
    if t is None:
        raise KeyError(f"no content/text field; keys are {sorted(rec)}")
    return t


def ws_grams(t, n=NGRAM):
    w = t.split()
    for i in range(len(w) - n + 1):
        yield " ".join(w[i:i + n])


def char_grams(t, n=NGRAM, stride=1):
    s = "".join(t.split())          # whitespace-insensitive, so formatting cannot hide a match
    for i in range(0, len(s) - n + 1, stride):
        yield s[i:i + n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt_p200m_4b_0902.pt")
    ap.add_argument("--heldout", default="data/sft/control_sft_text_heldout.jsonl")
    ap.add_argument("--ids", default="runs/heldout_v2/ids_shared.txt")
    ap.add_argument("--out", default="runs/e1_28_leak.json")
    ap.add_argument("--UNSAFE_skip_srcfp_check", action="store_true",
                    help="scan a domain even when its srcfp does not match the cursor's. ONLY for "
                         "the known-answer test, which necessarily builds a corpus the recorded "
                         "fingerprint cannot match. Stamps the output so a result produced this "
                         "way cannot be quoted as a measurement.")
    ap.add_argument("--root", default=None, help="tree holding data/corpus, for the "
                    "known-answer test. Defaults to the repo the script lives in; without this "
                    "the test's cwd was ignored and it scanned the real 232 GB corpus.")
    ap.add_argument("--limit_rows", type=int, default=0, help="0 = the cursor's count (the real "
                    "population); a small number is for testing the pipeline, never for a result")
    a = ap.parse_args()
    root = a.root or ROOT

    import torch
    ck = torch.load(os.path.join(root, a.ckpt), map_location="cpu", weights_only=False)
    cursor = ck.get("row_cursor")
    fps = ck.get("row_cursor_srcfp") or {}
    if not cursor:
        sys.exit(f"REFUSING: {a.ckpt} carries no row_cursor, so the rows it consumed are unknown "
                 f"and scanning data/corpus would measure a different population")
    print(f"cursor: {sum(cursor.values()):,} rows over {len(cursor)} domains, seed "
          f"{ck.get('row_cursor_seed')}")

    # 1. THE HELD-OUT COMPLETIONS, restricted to the scored population.
    keep = None
    if os.path.isfile(os.path.join(root, a.ids)):
        keep = {int(x) for x in open(os.path.join(root, a.ids)) if x.strip()}
        print(f"restricted to {len(keep):,} scored ids")
    targets = []          # (id, completion text)
    with open(os.path.join(root, a.heldout), errors="replace") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            # The record's OWN id field, int-coerced -- eval_heldout.py:85 uses d["id"] and
            # :513 matches int(x) from the ids file. My first version keyed on the LINE INDEX
            # and looked for a "completion" field; the fields are id/question/answer/src and the
            # id is a string. Two wrong guesses about one record shape, and the line-index
            # version would have silently scanned a shifted subset.
            if "id" not in r or "answer" not in r:
                sys.exit(f"REFUSING: heldout line {lineno} has keys {sorted(r)}; expected id and "
                         f"answer (eval_heldout.py:85 reads d['id'], d['question'], d['answer'])")
            rid = int(r["id"])
            if keep is not None and rid not in keep:
                continue
            # ONLY the answer: the question is masked in the loss (prompt tokens are -100), so a
            # question appearing in pretraining costs nothing on this metric. Scanning questions
            # too would report hits that cannot move either floor.
            targets.append((rid, r["answer"]))
    print(f"{len(targets):,} held-out completions to look for")
    if not targets:
        sys.exit("REFUSING: no completions -- an empty target set reports 0 hits and means nothing")

    # Build the needle sets ONCE, then stream the corpus past them.
    ws_need, ch_need = {}, {}
    for i, c in targets:
        for g in ws_grams(c):
            ws_need.setdefault(g, i)
        for g in char_grams(c, stride=7):
            ch_need.setdefault(g, i)
    print(f"needles: {len(ws_need):,} whitespace 13-grams, {len(ch_need):,} character 13-grams")

    # 2. STREAM THE CONSUMED ROWS.
    stale, results = [], {}
    for dom, nrows in sorted(cursor.items()):
        want = fps.get(dom)
        got, files = domain_fp(dom, root)
        if want and got != want and not a.UNSAFE_skip_srcfp_check:
            stale.append((dom, want, got, nrows))
            print(f"  {dom:<20} SRCFP CHANGED ({want} -> {got}) -- rows {nrows:,} are no longer "
                  f"the rows consumed; reported separately, NOT in the total")
            continue
        cap = a.limit_rows or nrows
        hits_ws, hits_ch, seen, chars = {}, {}, 0, 0
        use_char = dom in CHAR_WINDOW_DOMAINS
        for p in files:
            if seen >= cap:
                break
            with open(p, errors="replace") as f:
                for line in f:
                    if seen >= cap:
                        break
                    seen += 1
                    try:
                        t = text_of(json.loads(line))
                    except (KeyError, json.JSONDecodeError) as e:
                        sys.exit(f"REFUSING: {p} row {seen}: {e}")
                    chars += len(t)
                    for g in ws_grams(t):
                        if g in ws_need:
                            hits_ws.setdefault(ws_need[g], 0)
                            hits_ws[ws_need[g]] += 1
                    if use_char:
                        for g in char_grams(t, stride=1):
                            if g in ch_need:
                                hits_ch.setdefault(ch_need[g], 0)
                                hits_ch[ch_need[g]] += 1
        results[dom] = {"rows_scanned": seen, "rows_in_cursor": nrows, "chars": chars,
                        "srcfp": got, "ws_hit_ids": sorted(hits_ws), "char_hit_ids": sorted(hits_ch),
                        "char_window_applied": use_char}
        print(f"  {dom:<20} {seen:>9,} rows {chars / 1e9:>6.2f} GB  ws-hits "
              f"{len(hits_ws):>4}  char-hits {len(hits_ch) if use_char else '-':>4}")

    scanned = sum(r["rows_scanned"] for r in results.values())
    if scanned == 0:
        sys.exit(f"REFUSING: 0 rows were scanned ({len(stale)} domain(s) stale, "
                 f"{len(results)} usable) -- a scan that examined nothing would report "
                 f"'0 completions contaminated', which is the empty-population failure this "
                 f"repo has already shipped once. Nothing is concluded.")
    if scanned < 0.5 * sum(cursor.values()) and not a.limit_rows:
        print(f"WARNING: only {scanned:,} of {sum(cursor.values()):,} cursor rows were scanned "
              f"({len(stale)} domain(s) stale) -- any count below is a LOWER BOUND")

    all_ids = set()
    for r in results.values():
        all_ids |= set(r["ws_hit_ids"]) | set(r["char_hit_ids"])
    out = {
        "ckpt": a.ckpt,
        "rows_in_cursor": sum(cursor.values()),
        "rows_scanned": sum(r["rows_scanned"] for r in results.values()),
        "completions_checked": len(targets),
        "ngram": NGRAM,
        "char_window_domains": sorted(CHAR_WINDOW_DOMAINS),
        "per_domain": results,
        "stale_srcfp": [{"domain": d, "expected": w, "got": g, "rows": n} for d, w, g, n in stale],
        "contaminated_ids": sorted(all_ids),
        "limit_rows": a.limit_rows,
        "UNSAFE_skip_srcfp_check": a.UNSAFE_skip_srcfp_check,
    }
    with open(os.path.join(root, a.out), "w") as f:
        json.dump(out, f, indent=1)

    print(f"\n{len(all_ids)} of {len(targets):,} held-out completions appear in the rows this "
          f"checkpoint consumed")
    if a.UNSAFE_skip_srcfp_check:
        print("*** --UNSAFE_skip_srcfp_check WAS SET: the rows scanned are not verified to be "
              "the rows consumed. NOT A MEASUREMENT. ***")
    if a.limit_rows:
        print(f"*** limit_rows={a.limit_rows} -- THIS IS A PIPELINE TEST, NOT A RESULT ***")
    elif stale:
        print(f"*** {len(stale)} domain(s) have a changed srcfp and were NOT scanned; the count "
              f"above is a LOWER BOUND ***")
    elif not all_ids:
        print("ZERO on our side. The remaining unknown is entirely the Pile side, which cannot be "
              "scanned locally -- so the direction stops being unknown: any Pythia-side overlap "
              "would make its floor optimistic, i.e. would make the 2.00x floor gap an OVERstate.")
    else:
        print("Recompute the three verdict numbers on the clean subset with its own ids sha, and "
              "state that the other points were not recomputed.")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
