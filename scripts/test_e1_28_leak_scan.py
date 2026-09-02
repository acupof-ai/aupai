#!/usr/bin/env python3
"""The leak scanner finds a planted overlap, and does not invent one.

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

    # 2. A MISSING FIELD REFUSES rather than counting zero. This is the exact defect that made a
    #    six-domain probe print "no docs with >=13 whitespace tokens".
    try:
        S.text_of({"body": "x"})
        fails.append("text_of accepted a record with no content/text field")
    except KeyError as e:
        if "body" not in str(e):
            fails.append(f"text_of's error does not name the keys it saw: {e}")

    # 3. END TO END, with a PLANTED overlap and a control document that must NOT match.
    #    Run against a real checkpoint if one is here; otherwise this case reports SKIP, because
    #    a case that cannot run must say so rather than pass.
    ckpt = os.path.join(ROOT, "ckpt_p200m_4b_0902.pt")
    if not os.path.isfile(ckpt):
        SKIPS.append("case 3 (end-to-end, the planted-overlap check) needs "
                     "ckpt_p200m_4b_0902.pt, which is pod-only")
    else:
        with tempfile.TemporaryDirectory() as td:
            # A held-out file of two records. Record 7's answer is planted verbatim into the
            # corpus; record 8's is not.
            planted = ("the quick brown fox jumps over the lazy dog while seventeen blue "
                       "penguins recite arithmetic in the corridor")
            clean = ("an entirely different sentence about maritime navigation and the price "
                     "of copper in distant markets during autumn")
            ho = os.path.join(td, "ho.jsonl")
            with open(ho, "w") as f:
                for i, ans in ((7, planted), (8, clean)):
                    f.write(json.dumps({"id": str(i), "question": "q", "answer": ans,
                                        "src": "t"}) + "\n")
            ids = os.path.join(td, "ids.txt")
            open(ids, "w").write("7\n8\n")
            out = os.path.join(td, "out.json")

            # A fake corpus domain under --root, so nothing touches the real one. cwd is NOT
            # enough: the scanner derives its ROOT from __file__, so the first version of this
            # test silently began sha256-ing the real 232 GB corpus and hung with no output.
            fake_root = os.path.join(td, "root")
            os.makedirs(os.path.join(fake_root, "data", "corpus", "cot"))
            with open(os.path.join(fake_root, "data", "corpus", "cot", "a.jsonl"), "w") as f:
                f.write(json.dumps({"content": "preamble text " + planted + " trailing"}) + "\n")
                f.write(json.dumps({"content": "unrelated filler about geology"}) + "\n")

            try:
                r = subprocess.run(
                [sys.executable, os.path.join(ROOT, "scripts", "e1_28_leak_scan.py"),
                 "--ckpt", ckpt, "--heldout", ho, "--ids", ids, "--out", out,
                 "--root", fake_root, "--UNSAFE_skip_srcfp_check"],
                capture_output=True, text=True, timeout=120,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
            except subprocess.TimeoutExpired:
                fails.append("the scan did not finish in 120s against a 2-document fake corpus, "
                             "so it is not reading --root -- it is scanning something large")
                r = None
            blob = (r.stdout + r.stderr) if r else ""
            if r is None:
                pass
            elif "ModuleNotFoundError" in blob:
                SKIPS.append(f"case 3 (planted overlap): imports unavailable "
                             f"({blob.strip().splitlines()[-1][:80]})")
            elif not os.path.isfile(out):
                fails.append(f"end-to-end produced no output; tail: "
                             f"{blob.strip().splitlines()[-1][:140] if blob.strip() else '(silent)'}")
            else:
                d = json.load(open(out))
                hit = d.get("contaminated_ids", [])
                if 7 not in hit:
                    fails.append(f"THE PLANTED OVERLAP WAS NOT FOUND: contaminated_ids={hit}. A "
                                 f"scan that misses a verbatim 20-token match cannot be trusted "
                                 f"to report zero.")
                if 8 in hit:
                    fails.append(f"a clean completion was reported as contaminated: {hit}")

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
                elif "REFUSING: 0 rows were scanned" not in blob2:
                    fails.append(f"the scan failed but not with the zero-scanned refusal; "
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
    if not any("case 4" in x for x in SKIPS) and os.path.isfile(ckpt):
        ran.append("a scan that examined zero rows refuses instead of reporting zero hits")
    if any("case 3" in x for x in SKIPS):
        ran.append("BUT THE PLANTED-OVERLAP CHECK DID NOT RUN -- nothing here shows the scan can "
                   "find a real match, so a zero from it is not yet trustworthy")
    else:
        ran.append("a planted overlap is found while a clean record is not")
    print("e1_28_leak_scan OK: " + "; ".join(ran)
          + (f" [{len(SKIPS)} case(s) SKIPPED]" if SKIPS else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
