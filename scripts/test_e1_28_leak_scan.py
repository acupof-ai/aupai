#!/usr/bin/env python3
"""The leak scanner finds a planted overlap, and does not invent one.

# restartable: runs in-process assertions plus two subprocess scans over a 2-document fake corpus
# under a TemporaryDirectory. Seconds of CPU, no GPU, no state outside the temp dir. An interrupt
# costs the rerun and leaves nothing behind.

A containment scan that reports zero is indistinguishable from a broken one unless you have
planted a known answer. This repo has already shipped a leak check that returned "clean" on an
empty population and a probe that printed "no docs" for six domains because it read the wrong
field -- both were zeros that looked like findings.

    python3 scripts/test_e1_28_leak_scan.py
"""
import json
import os
import subprocess
import sys
import tempfile

SKIPS = []

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import e1_28_leak_scan as S  # noqa: E402


def main():
    fails = []

    # 1. THE UNITS. A whitespace 13-gram and a character 13-gram must both be produced, and the
    #    character one must ignore whitespace so reformatting cannot hide a match.
    txt = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi"
    ws = list(S.ws_grams(txt))
    if len(ws) != len(txt.split()) - 13 + 1:
        fails.append(f"ws_grams yielded {len(ws)} for {len(txt.split())} tokens")
    if any(len(g.split()) != 13 for g in ws):
        fails.append("a whitespace gram does not contain exactly 13 tokens")
    a = set(S.char_grams("abc def ghi jkl mno"))
    b = set(S.char_grams("abcdefghijklmno"))
    if a != b:
        fails.append("char_grams is whitespace-sensitive, so reformatting hides a match")
    if any(len(g) != 13 for g in a):
        fails.append("a character gram is not 13 characters")

    # 1b. THE LOW-ENTROPY FILTER drops markdown rules and keeps real text. A 13-char window over
    #     whitespace-stripped text matched '-------------' 538 times and '_____________' 66 times
    #     in the first full scan -- two different corpora reported the IDENTICAL 816 hits, which is
    #     a shared template, not leakage. The filter must kill those without touching content.
    for junk in ("-------------", "_____________", "-----|-------", "=============",
                 "  ---   ---  ", "aaaaaaaaaaaaa"):
        if not S.low_entropy("".join(junk.split()) or junk):
            fails.append(f"low_entropy passed a formatting gram: {junk!r}")
    for real in ("55个遗产地 2. 中国", "the quick brow", "def compute_x(", "意大利55个遗产地中国"):
        if S.low_entropy(real):
            fails.append(f"low_entropy DROPPED real content: {real!r} -- the filter is eating "
                         f"evidence, not noise")
    # THE BAND IS PINNED FROM BOTH SIDES, and the upper bound is a declared number rather than a
    # contrived string. My first attempt asserted that '沙沙沙沙沙沙沙沙响' and '1. 1. 1. 1. 1.'
    # must survive, to catch MIN_CHARSET=9 -- but those carry 2 and 3 distinct characters, so a
    # correct filter drops them, and they are precisely the "list skeleton" class this scan is
    # supposed to exclude. The test was wrong, not the filter. Asserting the bound directly says
    # what is actually being constrained: too low and markdown rules count as leakage, too high
    # and ordinary text gets discarded.
    if S.MIN_CHARSET < 2:
        fails.append(f"MIN_CHARSET={S.MIN_CHARSET} disables the filter")
    if S.MIN_CHARSET > 5:
        fails.append(f"MIN_CHARSET={S.MIN_CHARSET} is too aggressive: a 13-char window over "
                     f"ordinary text carries few distinct characters, so this discards content")

    # 1c. THE FILTER IS ACTUALLY APPLIED BY THE SCAN, not merely defined. Replacing the counting
    #     loop's `if low_entropy(g): continue` with `if False: continue` left every other case in
    #     this file green -- the filter could be disconnected entirely and nothing noticed, which
    #     is the same shape as a guard whose input is empty. A row carrying both a markdown rule
    #     and real content must produce filtered < raw, and must keep the real match.
    rule = "-" * 40
    # >= 13 whitespace tokens, or ws_grams yields nothing and the needle set is EMPTY -- which
    # made this case report "scan_text missed a verbatim whitespace match" against a correct
    # scanner. An empty needle set finds nothing and looks exactly like a broken scan.
    real = ("seventeen blue penguins recite arithmetic in a corridor at dawn today again "
            "and twice more before the harbour bell finally rings")
    ws_need = {g: 1 for g in S.ws_grams(real)}
    assert ws_need, "the fixture itself has no 13-gram; this case would test nothing"
    ch_need = {g: 2 for g in S.char_grams(rule)}
    for g in S.char_grams(real):
        ch_need.setdefault(g, 1)
    ws_h, ch_h, ch_raw = S.scan_text(f"prefix {rule} {real} suffix", ws_need, ch_need, True)
    if 1 not in ws_h:
        fails.append("scan_text missed a verbatim whitespace match")
    if 2 not in ch_raw:
        fails.append("the unfiltered character count lost the markdown rule, so 'raw' is not raw")
    if 2 in ch_h:
        fails.append("THE FILTER IS NOT WIRED INTO THE SCAN: a 40-dash markdown rule survived "
                     "into the filtered character hits. low_entropy() exists but the counting "
                     "loop does not call it")
    if 1 not in ch_h:
        fails.append("the filtered character count dropped a real content match")
    if len(ch_h) >= len(ch_raw):
        fails.append(f"filtered ({len(ch_h)}) is not smaller than raw ({len(ch_raw)}) on a row "
                     f"containing a markdown rule -- the two counts are the same number")
    _, ch_off, raw_off = S.scan_text(rule, ws_need, ch_need, False)
    if ch_off or raw_off:
        fails.append("use_char=False still produced character hits, so code domains would be "
                     "scanned with a unit the design excludes")

    # 2. A MISSING FIELD REFUSES rather than counting zero. This is the exact defect that made a
    #    six-domain probe print "no docs with >=13 whitespace tokens".
    try:
        S.text_of({"body": "x"})
        fails.append("text_of accepted a record with no content/text field")
    except KeyError as e:
        if "body" not in str(e):
            fails.append(f"text_of's error does not name the keys it saw: {e}")

    # 2b. THE FINGERPRINT MUST BE train.py's, NOT A LOOKALIKE. This case exists because my first
    #     version reimplemented it (sha256 of whole files) and reported SRCFP CHANGED for six of
     #     nine domains -- 720,000 rows including all of code_py_starcoder -- which I was about to
    #     report as "the corpus was rebuilt, those rows are unrecomputable". Two implementations of
    #     one quantity disagreeing is not a finding about the data.
    #
    #     Checked against the CHECKPOINT's own recorded values, which is the only witness that
    #     settles it: if the majority of domains disagree, the algorithm is wrong, not the corpus.
    ckpt_early = os.path.join(ROOT, "ckpt_p200m_4b_0902.pt")
    if not os.path.isfile(ckpt_early):
        SKIPS.append("case 2b (fingerprint parity with the checkpoint) needs "
                     "ckpt_p200m_4b_0902.pt, which is pod-only")
    else:
        try:
            import torch
            ck = torch.load(ckpt_early, map_location="cpu", weights_only=False)
            want = ck.get("row_cursor_srcfp") or {}
            agree, differ = [], []
            for dom, fp in sorted(want.items()):
                got, _ = S.domain_fp(dom)
                (agree if got == fp else differ).append(dom)
            if not want:
                SKIPS.append("case 2b: the checkpoint records no row_cursor_srcfp")
            elif len(differ) > len(agree):
                fails.append(f"{len(differ)} of {len(want)} domains disagree with the "
                             f"checkpoint's recorded srcfp ({', '.join(differ[:4])}...). With most "
                             f"domains disagreeing the fingerprint ALGORITHM is wrong, not the "
                             f"corpus -- import datagen/corpus_fingerprint.fp_dir, do not "
                             f"reimplement it")
            else:
                print(f"  case 2b: {len(agree)}/{len(want)} domains match the checkpoint's "
                      f"recorded srcfp"
                      + (f"; genuinely changed since: {', '.join(differ)}" if differ else ""),
                      file=sys.stderr)
        except ImportError as e:
            SKIPS.append(f"case 2b: {type(e).__name__}: {e}")

    # 3. THE CURSOR PATH MUST REFUSE, with the units statement.
    #    This case used to be the planted-overlap end-to-end check, and it was the case that would
    #    have caught the units bug if it could have run: it needs ckpt_p200m_4b_0902.pt, which is
    #    pod-only, so it reported SKIP on every local run while the scan's 316-item result was
    #    quoted in an audit. The scan now exits before touching the corpus, because its cursor
    #    restriction reads token-block rows as documents (5.0-13.2% of each domain). The
    #    planted-overlap capability lives in scripts/e1_28_heldout_contamination.py --selftest,
    #    which plants a span across three shards and runs everywhere with no checkpoint.
    ckpt = os.path.join(ROOT, "ckpt_p200m_4b_0902.pt")
    if not os.path.isfile(ckpt):
        SKIPS.append("case 3 (the cursor path refuses) needs ckpt_p200m_4b_0902.pt, which is "
                     "pod-only -- the refusal is unconditional in main(), so this only checks "
                     "that it fires before any corpus read")
    else:
        with tempfile.TemporaryDirectory() as td:
            ho = os.path.join(td, "ho.jsonl")
            open(ho, "w").write(json.dumps({"id": "7", "question": "q",
                                            "answer": " ".join(f"w{i}" for i in range(40)),
                                            "src": "t"}) + "\n")
            ids = os.path.join(td, "ids.txt")
            open(ids, "w").write("7\n")
            out = os.path.join(td, "out.json")
            fake_root = os.path.join(td, "root")
            os.makedirs(os.path.join(fake_root, "data", "corpus", "cot"))
            with open(os.path.join(fake_root, "data", "corpus", "cot", "a.jsonl"), "w") as f:
                f.write(json.dumps({"content": "filler"}) + "\n")
            try:
                r = subprocess.run(
                    [sys.executable, os.path.join(ROOT, "scripts", "e1_28_leak_scan.py"),
                     "--ckpt", ckpt, "--heldout", ho, "--ids", ids, "--out", out,
                     "--root", fake_root, "--UNSAFE_skip_srcfp_check"],
                    capture_output=True, text=True, timeout=120,
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
            except subprocess.TimeoutExpired:
                fails.append("the scan did not exit in 120s, so the refusal is not reached")
                r = None
            blob = (r.stdout + r.stderr) if r else ""
            if r is None:
                pass
            elif "ModuleNotFoundError" in blob:
                SKIPS.append(f"case 3: imports unavailable "
                             f"({blob.strip().splitlines()[-1][:80]})")
            elif r.returncode == 0:
                fails.append("the cursor path exited 0 instead of refusing -- its 5-13% document "
                             "coverage would be reported as the consumed population again")
            elif "WRONG UNITS" not in blob:
                fails.append(f"it refused, but without the units statement, so a reader cannot "
                             f"tell why; tail: {blob.strip().splitlines()[-1][:120]}")
            elif os.path.isfile(out):
                fails.append("it refused but still wrote an output file, which a later reader "
                             "would load as a measurement")

    # 4. A SCAN THAT EXAMINED NOTHING MUST REFUSE. Without --UNSAFE_skip_srcfp_check the fake
    #    corpus fails every srcfp check, so zero rows are scanned -- and the first version
    #    happily printed "0 of 2 completions appear" and wrote contaminated_ids: []. Zero
    #    scanned reporting zero found is the empty-population failure, not a clean result.
    if os.path.isfile(ckpt):
        with tempfile.TemporaryDirectory() as td:
            ho = os.path.join(td, "ho.jsonl")
            open(ho, "w").write(json.dumps({"id": "1", "question": "q",
                                            "answer": " ".join(f"w{i}" for i in range(40)),
                                            "src": "t"}) + "\n")
            ids = os.path.join(td, "ids.txt"); open(ids, "w").write("1\n")
            fr = os.path.join(td, "root")
            os.makedirs(os.path.join(fr, "data", "corpus", "cot"))
            open(os.path.join(fr, "data", "corpus", "cot", "a.jsonl"), "w").write(
                json.dumps({"content": "nothing relevant here at all"}) + "\n")
            try:
                r2 = subprocess.run(
                    [sys.executable, os.path.join(ROOT, "scripts", "e1_28_leak_scan.py"),
                     "--ckpt", ckpt, "--heldout", ho, "--ids", ids,
                     "--out", os.path.join(td, "o.json"), "--root", fr],
                    capture_output=True, text=True, timeout=120,
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
                blob2 = r2.stdout + r2.stderr
                if "ModuleNotFoundError" in blob2:
                    SKIPS.append("case 4 (zero-scanned refusal): imports unavailable")
                elif r2.returncode == 0:
                    fails.append("with every domain stale, the scan exited 0 instead of refusing "
                                 "-- it would report '0 contaminated' having examined no rows")
                elif "WRONG UNITS" not in blob2 and "REFUSING: 0 rows were scanned" not in blob2:
                    # The units refusal now fires FIRST and supersedes this one, so either message
                    # is a pass. Pinning only the old string would have gone red for the right
                    # behaviour -- a test asserting the exact refusal it was written against fails
                    # when an EARLIER, stronger refusal is added.
                    fails.append(f"the scan failed but with neither refusal; "
                                 f"tail: {blob2.strip().splitlines()[-1][:120]}")
            except subprocess.TimeoutExpired:
                fails.append("case 4 timed out, so --root is still not being honoured")

    for s_ in SKIPS:
        print(f"SKIP: {s_}", file=sys.stderr)
    for f in fails:
        print(f"FAIL: {f}", file=sys.stderr)
    if fails:
        return 1
    ran = ["both gram units are 13 wide", "character grams ignore whitespace",
           "a missing field refuses instead of counting zero"]
    if any("case 3" in x for x in SKIPS):
        ran.append("the cursor refusal is unconditional in main() but was not EXERCISED here "
                   "(no local checkpoint)")
    else:
        ran.append("the cursor path refuses with the units statement and writes no output")
    if not any("case 4" in x for x in SKIPS) and os.path.isfile(ckpt):
        ran.append("a scan that examined zero rows refuses instead of reporting zero hits")
    ran.append("the planted-overlap capability lives in e1_28_heldout_contamination.py --selftest, "
               "which needs no checkpoint and runs here")
    print("e1_28_leak_scan OK: " + "; ".join(ran)
          + (f" [{len(SKIPS)} case(s) SKIPPED]" if SKIPS else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
