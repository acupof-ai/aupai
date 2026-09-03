#!/usr/bin/env python3
"""RLVR data: build/load the verifiable-reward jsonl.

prepare() merges school_math_r1_zh (\\boxed{} answers) + gsm8k_zh (numeric
answers), normalizes, classifies each row as verifiable (single final value the
reward can score) or non-verifiable (solution-form aligned block / nested
multi-answer / unit-collapse-to-None), routes the latter to
rlvr_nonverifiable.jsonl at build (this repo's fail-loud principle: a derived
artifact must not silently carry rows the reward can never fire on), and
deduplicates the verifiable pool by prompt into data/rl/rlvr_math.jsonl
consumed by rlvr_trainer.py. classification uses the reward's OWN normalizer so
build and reward cannot disagree (the split-brain gap that let 米米 through).
Pure stdlib — importable without torch/GPU.

Usage: python algorithms/rlvr_data.py [--clean-audit]
  --clean-audit  run the build gate against the shipped rlvr_clean.jsonl and
                 print the routing volume (5 hard + 361 dead rows surface here
                 without touching the consumed artifact)
"""

import json
import os
import re
import sys

try:
    from .rlvr_reward import reward_fn, normalize_answer  # noqa: I001  (relative vs absolute import guard)
except ImportError:
    from rlvr_reward import reward_fn, normalize_answer  # noqa: I001

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
RLVR_PATH = os.path.join(DATA, "rl", "rlvr_math.jsonl")
NONVERIFIABLE_PATH = os.path.join(DATA, "rl", "rlvr_nonverifiable.jsonl")

# Solution-form scaffolding: a GT carrying these is a worked solution (aligned
# block / cases) or a multi-answer (nested \boxed). There is no single final
# value for the reward to fire on -- route to the non-verifiable pool.
_SOLUTION_ENV = re.compile(r"\\(begin|end)\{(aligned|cases|align|array)\}")
_MULTI_BOXED = re.compile(r"\\boxed\{", re.DOTALL)


def extract_boxed(text):
    """Extract answer from \\boxed{...} with balanced-brace matching."""
    results = []
    idx = 0
    while True:
        i = text.find("\\boxed{", idx)
        if i < 0:
            break
        depth = 1
        j = i + 7
        while j < len(text) and depth > 0:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
        if depth == 0:
            results.append(text[i + 7 : j - 1].strip())
        idx = j
    return results[-1] if results else None


def extract_gsm8k_answer(text):
    """Extract final numeric answer from GSM8K solution (last number)."""
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return nums[-1] if nums else None


def normalize(ans):
    """Normalize answer for comparison: strip LaTeX, whitespace, units."""
    if ans is None:
        return None
    s = str(ans).strip()
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)  # \text{cm} -> cm
    s = re.sub(r"\\dfrac\{([^}]+)\}\{([^}]+)\}", r"\1/\2", s)  # \dfrac{a}{b} -> a/b
    s = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"\1/\2", s)
    s = s.replace("\\", "").replace(" ", "").replace("（", "(").replace("）", ")")
    s = s.rstrip("。.,，")
    try:
        return str(float(s))
    except ValueError:
        return s


def load_problems(path=RLVR_PATH, want_meta=False):
    """Load prepared RLVR problems: [{prompt, answer, source}, ...].

    Data boundary: a row is refused unless its GT round-trips through the
    reward (reward_fn(\\boxed{gt}, gt) == 1.0) and its braces balance.
    Tolerance at the reward boundary is invisible; a refusal here is loud.

    THE _meta ROW IS PULLED OUT, not refused. It carries holdout_fp and the drop counts, and it
    has no `answer`, so leaving it in the loop would count it as one more unrewardable row --
    a stamp that inflates the refusal count by one is a stamp that made the data look worse.
    want_meta=True returns (problems, meta) so a caller can assert the stamp; meta is None for a
    file written before the stamp existed, and that difference is the point of reading it.
    """
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    meta = None
    if rows and "_meta" in rows[0]:
        meta = rows[0]["_meta"]
        rows = rows[1:]
    good, bad = [], 0
    for d in rows:
        gt = str(d.get("answer") or "")
        if gt.count("{") != gt.count("}") or reward_fn(f"\\boxed{{{gt}}}", gt) != 1.0:
            bad += 1
            continue
        good.append(d)
    if bad:
        print(
            f"[load_problems] REFUSED {bad}/{len(rows)} rows from {path}: "
            f"GT does not round-trip through reward_fn (unbalanced braces or "
            f"unrewardable answer form)",
            file=sys.stderr,
            flush=True,
        )
    return (good, meta) if want_meta else good


def classify_verifiable(ans):
    """Route a raw extracted answer: VERIFIABLE (a single final value the reward
    can score) or NONVERIFIABLE.

    A row is non-verifiable (route to the pool, fail-loud at build) if:
      - its answer is solution-form (\\begin{aligned}/cases/align) -- no single
        final value to fire on;
      - it nests a multiple-answer \\boxed{} (\\boxed within \\boxed);
      - reward's normalizer collapses a non-empty answer to None -- the unit
        strip ate a name/word (the 米米 class). Detected here, not silently
        passed: build and reward must share one normalizer, or build admits
        rows reward refuses (the split-brain gap).
    """
    if ans is None:
        return "NONVERIFIABLE"
    if _SOLUTION_ENV.search(ans) is not None:
        return "NONVERIFIABLE"
    if ans.count(r"\boxed") > 1:
        return "NONVERIFIABLE"
    # the reward's own normalizer is the source of truth for scoring
    if normalize_answer(ans) is None:
        return "NONVERIFIABLE"
    return "VERIFIABLE"


HELDOUT_PATH = os.path.join(DATA, "rl", "rlvr_heldout_dropped.jsonl")


def _holdout_filter():
    """(is_holdout, fp) -- THE SHARED predicate and the fingerprint of the set it read.

    IMPORTED from datagen.holdout, never reimplemented: is_holdout(q) is the same function the
    corpus build and the SFT pack use, so a question this repo calls held out cannot be held out
    for one artifact and not another. Reimplementing the hash here is the §145 shape -- two
    definitions under one name, agreeing until the day they do not.

    Returns (None, None) when the holdout file is absent. The CALLER decides what that means; a
    silent pass would be the exact failure this repo already paid for (an empty holdout_hashes let
    19 of 20 questions into SFT), so prepare() refuses instead.
    """
    import hashlib

    sys.path.insert(0, ROOT)
    try:
        from datagen.holdout import HASH_PATH, is_holdout
    except ImportError:
        return None, None
    if not os.path.exists(HASH_PATH):
        return None, None
    fp = hashlib.sha256(open(HASH_PATH, "rb").read()).hexdigest()[:16]
    return is_holdout, fp


def prepare(data_dir=DATA, out_path=RLVR_PATH, nonverif_path=NONVERIFIABLE_PATH,
            heldout_path=HELDOUT_PATH, require_holdout=True):
    """Build the verifiable RLVR jsonl + the routed non-verifiable pool from the
    raw datasets. Returns (verifiable_items, nonverifiable_items).

    HELD-OUT QUESTIONS ARE DROPPED, not carried, and the dropped rows are WRITTEN OUT so the
    volume is auditable rather than a count in a log. Measured 2026-09-03 on the first build:
    515 of 218,095 rows (0.2361%) were holdout questions. RL is the worst place to leak a
    holdout question -- the reward fires on the answer, so a leaked row is trained toward the
    exact value the eval scores, and the eval then reads as capability. This is not a
    hypothetical: the three sample hits were school_math rows whose text is verbatim eval text.
    """
    out = []
    nonverif = []
    heldout = []
    is_holdout, holdout_fp = _holdout_filter()
    if is_holdout is None:
        if require_holdout:
            raise RuntimeError(
                "data/eval/holdout_hashes.txt is unreadable, so this build cannot say which rows "
                "are held out. REFUSING rather than writing an unfiltered pool: an absent holdout "
                "set that reads as 'nothing is held out' is how 19 of 20 questions reached SFT. "
                "Pass require_holdout=False only to build a pool you will not train on.")
        holdout_fp = "UNFILTERED"

    def _emit(d, ans, source):
        # THE HOLDOUT DROP COMES FIRST, before the verifiable/non-verifiable split: a held-out
        # question must not reach either pool. Routing it to nonverifiable would keep it on disk
        # under a name that reads as "safe to look at".
        if is_holdout is not None and is_holdout(d["instruction"]):
            heldout.append({"prompt": d["instruction"], "answer": ans, "source": source,
                            "route": "HOLDOUT"})
            return
        if classify_verifiable(ans) == "VERIFIABLE":
            out.append({"prompt": d["instruction"], "answer": ans, "source": source})
        else:
            nonverif.append({"prompt": d["instruction"], "answer": ans, "source": source, "route": classify_verifiable(ans)})

    # school_math_r1_zh: 223K problems with \boxed{} answers
    with open(os.path.join(data_dir, "school_math_r1_zh.jsonl"), encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            ans = extract_boxed(d["output"])
            if ans and normalize(ans):
                _emit(d, ans, "school_math")

    # gsm8k_zh: 7.5K problems with numeric answers
    with open(os.path.join(data_dir, "gsm8k_zh.jsonl"), encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            ans = extract_gsm8k_answer(d["output"])
            if ans and normalize(ans):
                _emit(d, ans, "gsm8k")

    seen = set()
    deduped = []
    for item in out:
        if item["prompt"] not in seen:
            seen.add(item["prompt"])
            deduped.append(item)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        # THE STAMP IS THE FIRST LINE, and it is a row rather than a sidecar file so it cannot be
        # separated from the data it describes. load_problems skips it by the "_meta" key.
        # Without this the artifact is indistinguishable from one built before the filter existed
        # -- exactly the state twelve of sixteen SFT packs are in today (holdout_fp absent, which
        # sft_math.py accepts with a WARNING), and an absent stamp reads as "no holdout applied"
        # for a file that may or may not have had it.
        f.write(json.dumps({"_meta": {"holdout_fp": holdout_fp,
                                      "heldout_dropped": len(heldout),
                                      "verifiable": len(deduped),
                                      "nonverifiable": len(nonverif)}},
                           ensure_ascii=False) + "\n")
        for item in deduped:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    with open(nonverif_path, "w", encoding="utf-8") as f:
        for item in nonverif:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    with open(heldout_path, "w", encoding="utf-8") as f:
        for item in heldout:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return deduped, nonverif


def audit_clean(clean_path):
    """One-off: run the build gate against the shipped rlvr_clean.jsonl and
    report the routing volume. The clean artifact predates the gate; its
    dead rows (solution-form / unit-collapse) currently pass load_problems
    silently. This audit surfaces them without touching the consumed file."""
    clean_path = clean_path or os.path.join(DATA, "rl", "rlvr_clean.jsonl")
    counts = {"verifiable": 0, "solution_form": 0, "multi_boxed": 0, "unit_collapse": 0, "empty": 0}
    n = 0
    with open(clean_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            ans = str(d.get("answer") or "")
            n += 1
            if not ans:
                counts["empty"] += 1
                continue
            if _SOLUTION_ENV.search(ans) is not None:
                counts["solution_form"] += 1
                continue
            if ans.count(r"\boxed") > 1:
                counts["multi_boxed"] += 1
                continue
            if normalize_answer(ans) is None:
                counts["unit_collapse"] += 1
                continue
            counts["verifiable"] += 1
    return n, counts


def _selftest():
    """The holdout drop and the stamp, on a fixture with a known answer. No data, no card.

    Only the two things this change added are tested here. The verifiable/non-verifiable routing
    already had its coverage, and re-asserting it would test a second copy of those assertions.
    """
    import tempfile

    d = tempfile.mkdtemp()
    src = {"held": "HELD QUESTION", "keep": "KEPT QUESTION"}
    os.makedirs(os.path.join(d, "rl"), exist_ok=True)
    for name, q in (("school_math_r1_zh.jsonl", src["held"]), ("gsm8k_zh.jsonl", src["keep"])):
        with open(os.path.join(d, name), "w", encoding="utf-8") as f:
            if name.startswith("school"):
                f.write(json.dumps({"instruction": q, "output": r"so \boxed{7}"}) + "\n")
                f.write(json.dumps({"instruction": "ALSO KEPT", "output": r"so \boxed{9}"}) + "\n")
            else:
                f.write(json.dumps({"instruction": q, "output": "answer is 5\n#### 5"}) + "\n")

    out_p = os.path.join(d, "rl", "out.jsonl")
    nv_p = os.path.join(d, "rl", "nv.jsonl")
    ho_p = os.path.join(d, "rl", "ho.jsonl")

    # A STUB PREDICATE holding exactly one question, so the expected drop count is known
    # independently of the real holdout file (which changes) -- the fixture must not be able to
    # pass by agreeing with whatever the live set happens to contain.
    # THIS module object, whichever name it was imported under: run as a script it is __main__,
    # imported it is algorithms.rlvr_data. The patch must land on the object prepare() actually
    # reads its filter from, or the stub is installed on a second copy and the fixture tests
    # nothing while passing.
    R = sys.modules[__name__]
    real = R._holdout_filter
    R._holdout_filter = lambda: ((lambda q: q == src["held"]), "fixture_fp")
    try:
        verif, nonverif = prepare(data_dir=d, out_path=out_p, nonverif_path=nv_p, heldout_path=ho_p)
        prompts = {v["prompt"] for v in verif}
        assert src["held"] not in prompts, "a HOLDOUT question reached the verifiable pool"
        assert src["keep"] in prompts and "ALSO KEPT" in prompts, prompts
        # THE DROPPED ROW IS ON DISK, not just absent: a count in a log cannot be audited later.
        dropped = [json.loads(x) for x in open(ho_p, encoding="utf-8") if x.strip()]
        assert len(dropped) == 1 and dropped[0]["prompt"] == src["held"], dropped
        assert dropped[0]["route"] == "HOLDOUT", dropped[0]
        # THE STAMP IS THE FIRST ROW and load_problems pulls it out rather than refusing it.
        probs, meta = load_problems(out_p, want_meta=True)
        assert meta and meta["holdout_fp"] == "fixture_fp", meta
        assert meta["heldout_dropped"] == 1, meta
        assert len(probs) == len(verif), (len(probs), len(verif), "the _meta row was counted as a problem")
        assert all("_meta" not in p for p in probs), "the stamp row leaked into the problem list"
        # A FILE WITH NO STAMP returns meta None rather than reading row 0 as a stamp.
        plain = os.path.join(d, "rl", "plain.jsonl")
        with open(plain, "w", encoding="utf-8") as f:
            f.write(json.dumps({"prompt": "q", "answer": "7", "source": "x"}) + "\n")
        p2, m2 = load_problems(plain, want_meta=True)
        assert m2 is None and len(p2) == 1, (m2, p2)
    finally:
        R._holdout_filter = real

    # AN UNREADABLE HOLDOUT SET REFUSES rather than writing an unfiltered pool. This is the
    # failure mode that let 19 of 20 questions into SFT: absent read as "nothing is held out".
    R._holdout_filter = lambda: (None, None)
    try:
        prepare(data_dir=d, out_path=out_p, nonverif_path=nv_p, heldout_path=ho_p)
        raise AssertionError("an unreadable holdout set was accepted")
    except RuntimeError as e:
        assert "REFUSING" in str(e), e
    finally:
        R._holdout_filter = real

    print("rlvr_data self-test OK: a holdout question is dropped from BOTH pools and written to "
          "the dropped file with route=HOLDOUT; the stamp is row 0 carrying holdout_fp and the "
          "drop count; load_problems pulls the stamp out instead of counting it as an unrewardable "
          "row, and returns meta None for an unstamped file; and an unreadable holdout set refuses "
          "instead of writing an unfiltered pool")


def main():
    if "--selftest" in sys.argv:
        _selftest()
        return
    if "--clean-audit" in sys.argv:
        n, c = audit_clean(None)
        print(
            f"audit of rlvr_clean.jsonl: {n} rows -> "
            f"verifiable {c['verifiable']} | routing {n - c['verifiable']} "
            f"(solution_form {c['solution_form']}, multi_boxed {c['multi_boxed']}, "
            f"unit_collapse {c['unit_collapse']}, empty {c['empty']})"
        )
        return
    verif, nonverif = prepare()
    print(
        f"Total verifiable: {len(verif)} problems "
        f"(school_math: {sum(1 for d in verif if d['source'] == 'school_math')}, "
        f"gsm8k: {sum(1 for d in verif if d['source'] == 'gsm8k')})"
    )
    print(f"Routed non-verifiable: {len(nonverif)} -> {NONVERIFIABLE_PATH}")


if __name__ == "__main__":
    main()
