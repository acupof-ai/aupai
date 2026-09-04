#!/usr/bin/env python3
"""Audit pass 2: recompute fact VALUES against their cited artifacts, seed 904.

Pass 1 (audit_fact_sources.py) asked whether a fact's source EXISTS. This asks the harder
question the charter actually wants: does the number in the fact appear in the artifact it
cites. A fact whose artifact exists and whose number is wrong passes pass 1 silently.

THE SAMPLE MUST BE REPRODUCIBLE OR IT IS NOT A SAMPLE. Drawn from `sorted(ids)` — one
canonical order, built once here — then `random.seed(904); random.sample(...)`. My first
two attempts at this drew DIFFERENT samples because each rebuilt the population list a
slightly different way (one sorted by (file, id), the other by id alone), which is the same
class of defect as the tokeniser problem in pass 1: the enumeration step, not the checking
step. Seed 904 matches 44's so the two samples can be compared for overlap.

WHAT "RECOMPUTED" MEANS HERE, stated because a weaker check would look identical. For each
sampled fact this extracts every distinct number from `value` and searches the cited
on-disk artifacts for it, at the artifact's own precision. Three outcomes:

    HELD      every number in the value that the artifact could carry was found there
    PARTIAL   some found, some not -- the not-found ones are printed, since a derived
              number (a ratio, a median) legitimately does not appear in a raw log
    NO-FILE   no cited artifact is readable here (pass 1's finding, restated per fact)

This is a TEXT search, and that ceiling is the honest limit: it proves a number appears in
the artifact, not that the artifact's own arithmetic is right. A median quoted correctly
from a log whose lines are wrong reads as HELD. What it does catch is the case the charter
is aimed at -- a published number that is in no artifact at all.

BROKEN-WORLD TEST: --selftest builds a fact whose value holds a number its log does not
contain, and asserts this reports it. An instrument that says HELD for everything has not
run.
"""
import json
import os
import random
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FILES = ["facts/efficiency.json", "facts/smelt_deeploop.json"]
SEED = 904

# A number worth checking. TWO DEFECTS ITS OWN SELFTEST CAUGHT, both in the boundaries:
#   trailing \b killed "11.87K" -- digit followed by a letter is NOT a word boundary, so
#     every K/M/G-suffixed figure in this repo (most throughput numbers) matched nothing;
#   \b\d{3,}(?:,\d{3})* matched "189,548" inside "1,189,548" -- the leading group is
#     1-3 digits, so a 7-digit grouped number truncates and the search then looks for a
#     number the artifact does not contain, reporting PARTIAL on a sound fact.
# Now: a decimal, or a comma-grouped integer anchored at its true start, or a bare run of
# 3+ digits. Leading (?<![\d,.]) anchors the start; no trailing \b.
NUM = re.compile(r"(?<![\d,.])\d{1,3}(?:,\d{3})+|(?<![\d,.])\d+\.\d+|(?<![\d,.])\d{3,}")
# Paths in a source field, same discipline as pass 1 but simpler: extension-bearing tokens.
PATH = re.compile(r"[\w./\-]+\.(?:log|jsonl|json|txt|md|py|sh)\b")


def load_facts(root=ROOT, files=None):
    out = {}
    for f in (files or FILES):
        for fact in json.load(open(os.path.join(root, f), encoding="utf-8"))["facts"]:
            out[fact["id"]] = fact
    return out


def draw(ids, seed=SEED, n=40):
    """The ONE canonical draw. sorted() first so the population order cannot vary."""
    pop = sorted(ids)
    rng = random.Random(seed)
    return sorted(rng.sample(pop, min(n, len(pop))))


def artifacts_for(fact, root=ROOT):
    """Readable on-disk artifacts this fact cites, from source AND config/uncertainty --
    58's sweep found four paths mine missed by reading source only."""
    blob = " ".join(str(fact.get(k, "")) for k in
                    ("source", "config", "uncertainty", "boundary"))
    if isinstance(fact.get("config"), dict):
        blob += " " + json.dumps(fact["config"], ensure_ascii=False)
    hits = []
    for m in dict.fromkeys(PATH.findall(blob)):
        p = os.path.join(root, m.lstrip("/"))
        if os.path.isfile(p) and os.path.getsize(p) < 40 * 1024 * 1024:
            hits.append(m)
    return hits


def nums(s):
    return list(dict.fromkeys(NUM.findall(str(s))))


def check(fact, root=ROOT):
    arts = artifacts_for(fact, root)
    want = nums(fact.get("value", ""))
    if not arts:
        return "NO-FILE", want, []
    text = ""
    for a in arts:
        try:
            text += open(os.path.join(root, a.lstrip("/")), encoding="utf-8",
                         errors="replace").read()
        except OSError:
            pass
    # PRECISION IS NOT DISAGREEMENT, and treating it as such made this instrument useless:
    # the first run reported 22 of 40 PARTIAL, and the first one I checked by hand was
    # repo.loop_from_scratch_stage_d "missing" 122.30 while its log prints `params 122.3M`.
    # Same number, fewer digits. A fact legitimately quotes more precision than a log line
    # (it was computed from the tensors, not read off the print) and legitimately quotes
    # less (a rounded summary). So a number counts as found if the artifact holds it at
    # ANY precision that does not contradict it: strip trailing zeros, then look for the
    # fact's digits as a prefix of a number in the text, or the text's digits as a prefix
    # of the fact's. What still fails is a genuinely different value.
    flat = text.replace(",", "")
    def present(w):
        raw = w.replace(",", "")
        if raw in flat:
            return True
        if "." not in raw:
            return False
        head = raw.rstrip("0").rstrip(".")
        if head and head in flat:                      # 1.0670 vs 1.067
            return True
        # The log is coarser: it prints 122.3 where the fact says 122.30. Compare NUMERICALLY
        # at the artifact's own precision instead of truncating the wanted string -- the
        # truncating version accepted 11.9 against a log reading 11.87, because it cut the
        # wanted number back to "11." and found that. Its own fixture caught it.
        try:
            wv = float(raw)
        except ValueError:
            return False
        for cand in NUM.findall(flat):
            c = cand.replace(",", "")
            if "." not in c:
                continue
            dp = len(c.split(".")[1])
            try:
                if round(wv, dp) == float(c):       # 122.30 -> 122.3 at 1 dp: equal
                    return True
            except ValueError:
                continue
        return False
    found = [w for w in want if present(w)]
    missing = [w for w in want if w not in found]
    if not want:
        return "NO-NUM", [], arts
    return ("HELD" if not missing else "PARTIAL"), missing, arts


def selftest():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "runs"))
        open(os.path.join(d, "runs", "a.log"), "w").write("median 11.87K over 190 lines\n")
        os.makedirs(os.path.join(d, "facts"))
        json.dump({"facts": [
            {"id": "t.held", "value": "median 11.87K", "source": "runs/a.log"},
            {"id": "t.wrong", "value": "median 99.99K", "source": "runs/a.log"},
            {"id": "t.nofile", "value": "12.34 things", "source": "runs/gone.log"},
            # precision tolerance must NOT swallow a real disagreement: the log says
            # 11.87 and this claims 11.9, which rounds the other way. Wanted-as-prefix
            # would accept 11.8; 11.9 must still fail.
            {"id": "t.coarse_ok", "value": "median 11.870", "source": "runs/a.log"},
            {"id": "t.coarse_bad", "value": "median 11.9", "source": "runs/a.log"},
        ]}, open(os.path.join(d, "facts", "t.json"), "w"))
        f = load_facts(d, ["facts/t.json"])
        assert check(f["t.held"], d)[0] == "HELD", check(f["t.held"], d)
        v, miss, _ = check(f["t.wrong"], d)
        assert v == "PARTIAL" and miss == ["99.99"], (v, miss)
        assert check(f["t.nofile"], d)[0] == "NO-FILE"
        # 11.870 IS 11.87 (trailing zero) -- must hold
        assert check(f["t.coarse_ok"], d)[0] == "HELD", check(f["t.coarse_ok"], d)
        # 11.9 is NOT 11.87 -- tolerance must not accept it
        v2, miss2, _ = check(f["t.coarse_bad"], d)
        assert v2 == "PARTIAL" and miss2 == ["11.9"], (v2, miss2)
        # and the draw must be stable across calls
        a = draw([f"id{i}" for i in range(50)], n=10)
        b = draw([f"id{i}" for i in range(50)], n=10)
        assert a == b, "draw is not reproducible"
    print("selftest OK: catches a number absent from its artifact, distinguishes NO-FILE, "
          "and the seed-904 draw is stable across calls")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest(); raise SystemExit(0)
    facts = load_facts()
    sample = draw(list(facts), n=40)
    print(f"population {len(facts)} facts; sample {len(sample)} at seed {SEED}\n")
    tally = {}
    for fid in sample:
        verdict, missing, arts = check(facts[fid])
        tally[verdict] = tally.get(verdict, 0) + 1
        line = f"  {verdict:8s} {fid:56s}"
        if verdict == "PARTIAL":
            line += f" missing {missing[:6]}"
        elif verdict == "HELD":
            line += f" ({len(arts)} artifact(s))"
        print(line)
    print("\n" + "  ".join(f"{k}={v}" for k, v in sorted(tally.items())))
