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
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHAR_WINDOW_DOMAINS = {"zh_web", "chatml", "chat_qa"}
NGRAM = 13

# A 13-CHARACTER WINDOW OVER LOW-ENTROPY TEXT MATCHES FORMATTING, NOT CONTENT.
# The first full scan reported chat_qa 816 char-hits and chatml 816 char-hits -- the SAME count
# from two different corpora, which is the signature of a shared template rather than leakage.
# Printing the matched strings settled it: '-------------' x538, '_____________' x66, and table
# rules like '-----|-------' x30. Thirteen characters of a markdown rule collide by necessity.
# So a character gram drawn from fewer than this many distinct characters is not evidence. The
# threshold is declared here, BEFORE the numbers, and both the filtered and unfiltered counts are
# reported -- a filter introduced after seeing the hits and reported only post-filter is a knob
# tuned to make a number look better.
MIN_CHARSET = 4

# Distinct matched grams recorded per domain, so the whitespace hits can be read by a human
# instead of reasoned about from a count. Truncation is recorded in the output: a silent cap
# reads as "this was all of them".
MAX_RECORDED = 400


def low_entropy(g):
    """True when a gram is built from too few distinct characters to be evidence of anything."""
    return len(set(g)) < MIN_CHARSET


def scan_text(t, ws_need, ch_need, use_char):
    """One row's matches: (whitespace ids, character ids kept, character ids before the filter).

    This is a function rather than three inline loops because the filter's APPLICATION needs a
    test, not just the predicate. With low_entropy() tested on its own, replacing the loop's
    `if low_entropy(g): continue` with `if False: continue` left every case green -- the filter
    could be disconnected from the scan entirely and nothing noticed. A predicate nobody calls is
    not a filter.

    SETS, not counters. These were dicts of id -> match count, but only the KEYS ever reached the
    output (`sorted(hits_ws)`), so deleting `count += 1` changed nothing any test or any result
    could see -- three mutations of the counting lines passed a green suite because the value was
    dead. Per-gram counts are still kept, in matched_*_grams, where they are actually read.
    """
    ws, ch, ch_raw = set(), set(), set()
    for g in ws_grams(t):
        if g in ws_need:
            ws.add(ws_need[g])
    if use_char:
        for g in char_grams(t, stride=1):
            if g in ch_need:
                # BOTH sets. ch_raw is every match; ch drops the grams that cannot be evidence
                # (markdown rules), per MIN_CHARSET above.
                ch_raw.add(ch_need[g])
                if not low_entropy(g):
                    ch.add(ch_need[g])
    return ws, ch, ch_raw


def domain_fp(d, root=None):
    """The CANONICAL fingerprint, imported -- never reimplemented.

    row_cursor_srcfp is written by train.py:1796 from _corpus_fp, which is sha1 over sorted
    "name:size:sha256(first 64KB):sha256(last 64KB)" per shard, skipping build_corpus_stats.json.
    My first version invented its own: sha256 of every full file, no skip list. It reported
    SRCFP CHANGED for six of nine domains -- 720,000 rows including all of code_py_starcoder --
    and I was one message away from reporting that as "the corpus was rebuilt since that run and
    those rows are no longer recomputable". Every one of those was my hash disagreeing with
    train.py's, which is what comparing two implementations of one quantity produces.
    datagen/corpus_fingerprint.py's own --self-check exists to assert parity with train.py, so
    fp_dir is the only honest source here.
    """
    base = os.path.join(root or ROOT, "data", "corpus", d)
    files = sorted(glob.glob(os.path.join(base, "*.jsonl")))
    if not os.path.isdir(base):
        return None, files
    sys.path.insert(0, os.path.join(ROOT, "datagen"))
    from corpus_fingerprint import fp_dir
    return fp_dir(base), files


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
    sys.exit(
        "REFUSING: this scanner's cursor restriction is in the WRONG UNITS and its result "
        "cannot be quoted.\n"
        "row_cursor counts TOKEN-BLOCK rows (cache.numel()//(seq+1), see mix_200m_4b.json's "
        "epochs_pool_source), and the loop below counts JSONL DOCUMENTS -- `for line in f: "
        "seen += 1` capped at the cursor value. On ckpt_p200m_4b_0902 that reads 5.0-13.2% of "
        "each domain's documents and reports it as 'the rows this run consumed': math_owm 7.6%, "
        "en_c4 5.0%, cot 11.2%, textbook 13.2%, chatml 5.5%, chat_qa 5.5%, zh_web 0.3%, "
        "starcoder 6.4%, rp1t 9.1%.\n"
        "The `scanned < 0.5 * sum(cursor)` warning below cannot catch this: it compares the "
        "mis-unit'd count against the cap it was taken from, so it is in the same units as the "
        "bug and never fires.\n"
        "MEASURED CONSEQUENCE: this scan reported 316 contaminated items (runs/e1_28_matched.json, "
        "312 ws + 4 universal-only) and docs/audits/control_pythia160m_vs_ours.md 5.3d re-scored "
        "the floors on that exclusion. A whole-corpus scan of the same population found 2,114 of "
        "7,523 measurable (28.10%): scripts/e1_28_heldout_contamination.py, "
        "runs/e1_28_heldout_contamination.json. On chatml alone this saw 8,778 of 160,414 "
        "documents and found 40; the full scan found 1,515.\n"
        "It also SAW the signal and filtered it: chatml and chat_qa both reported 808 char-hits, "
        "and the docstring above names an identical count from two corpora as the signature of a "
        "shared template. It is -- they are one source rendered twice (build_chatml.py and "
        "build_chat_qa.py over data/corpus/chat).\n"
        "WHAT TO USE: scripts/e1_28_heldout_contamination.py for a corpus-wide containment rate. "
        "This file stays because datagen/scan_eval_golds.py imports ws_grams/char_grams/scan_text/"
        "low_entropy from it -- the GRAM FUNCTIONS are sound and tested; only the cursor "
        "population is not. A consumed-rows rate needs a document-level cursor, which no "
        "instrument here provides, and the pre-tokenize shuffle at train.py:1456 means the first "
        "N documents are not the first N of the token cache either.")

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
        hits_ws, hits_ch, hits_ch_raw = set(), set(), set()
        seen, chars = 0, 0
        grams_ws, grams_ch = {}, {}
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
                    ws_h, ch_h, ch_r = scan_text(t, ws_need, ch_need, use_char)
                    hits_ws |= ws_h
                    hits_ch |= ch_h
                    hits_ch_raw |= ch_r
                    # The distinct grams themselves, so a human reads the strings rather than
                    # reasoning from a count -- which is how the 816 formatting hits were caught.
                    for g in ws_grams(t):
                        if g in ws_need and (g in grams_ws or len(grams_ws) < MAX_RECORDED):
                            grams_ws[g] = grams_ws.get(g, 0) + 1
                    if use_char:
                        for g in char_grams(t, stride=1):
                            if g in ch_need and not low_entropy(g) and (
                                    g in grams_ch or len(grams_ch) < MAX_RECORDED):
                                grams_ch[g] = grams_ch.get(g, 0) + 1
        results[dom] = {"rows_scanned": seen, "rows_in_cursor": nrows, "chars": chars,
                        "srcfp": got, "ws_hit_ids": sorted(hits_ws),
                        "char_hit_ids": sorted(hits_ch),
                        "char_hit_ids_unfiltered": sorted(hits_ch_raw),
                        "char_window_applied": use_char,
                        "matched_ws_grams": grams_ws, "matched_char_grams": grams_ch,
                        "matched_grams_truncated": (len(grams_ws) >= MAX_RECORDED
                                                    or len(grams_ch) >= MAX_RECORDED)}
        print(f"  {dom:<20} {seen:>9,} rows {chars / 1e9:>6.2f} GB  ws-hits "
              f"{len(hits_ws):>4}  char-hits "
              f"{(str(len(hits_ch)) + '/' + str(len(hits_ch_raw))) if use_char else '-':>9}"
              f"{'  (filtered/raw)' if use_char else ''}")

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
    raw_ids = set()
    for r in results.values():
        all_ids |= set(r["ws_hit_ids"]) | set(r["char_hit_ids"])
        raw_ids |= set(r["ws_hit_ids"]) | set(r["char_hit_ids_unfiltered"])
    out = {
        "ckpt": a.ckpt,
        "rows_in_cursor": sum(cursor.values()),
        "rows_scanned": sum(r["rows_scanned"] for r in results.values()),
        "completions_checked": len(targets),
        "ngram": NGRAM,
        "char_window_domains": sorted(CHAR_WINDOW_DOMAINS),
        "char_min_charset": MIN_CHARSET,
        "per_domain": results,
        "stale_srcfp": [{"domain": d, "expected": w, "got": g, "rows": n} for d, w, g, n in stale],
        "contaminated_ids": sorted(all_ids),
        "contaminated_ids_unfiltered": sorted(raw_ids),
        "limit_rows": a.limit_rows,
        "UNSAFE_skip_srcfp_check": a.UNSAFE_skip_srcfp_check,
    }
    with open(os.path.join(root, a.out), "w") as f:
        json.dump(out, f, indent=1)

    print(f"\n{len(all_ids)} of {len(targets):,} held-out completions appear in the rows this "
          f"checkpoint consumed ({len(raw_ids)} before the MIN_CHARSET={MIN_CHARSET} filter on "
          f"character grams)")
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
