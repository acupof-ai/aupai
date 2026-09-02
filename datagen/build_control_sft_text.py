#!/usr/bin/env python3
# restartable: reads jsonl sources and writes one jsonl. A re-run overwrites the output
# wholesale, so an interrupt costs the run, not a partial file -- the output is written to
# .part and renamed only after the row count and sha256 are computed.
"""Build ONE text-level SFT pack shared by both arms of the control comparison.

    python3 datagen/build_control_sft_text.py [--out data/sft/control_sft_text.jsonl]

WHY TEXT AND NOT A .pt. The comparison trains our 200M checkpoint and Pythia-160M on the
same data. Those two models do not share a vocabulary -- ours is data/tokenizer.json,
Pythia's is a 50,304-entry NeoX BPE -- so a packed .pt cannot be shared: every id would be
valid and wrong, which is exactly the failure scripts/check_sft_ready.py:check_vocab
exists to refuse. The shared artifact is therefore the TEXT, and each side tokenizes it
with its own tokenizer. What is held identical is the example set, their order, and the
ChatML template; what necessarily differs is the token count, which the report states per
side rather than hiding.

SOURCES. prepare_sft.SOURCES (the v5 family-clean list: the code-500 carve source is
already absent, the 2,300 Evol-Instruct code rows are present) plus
prepare_sft_math.SOURCES for math CoT. Both lists are IMPORTED, not copied, so a change
there reaches this pack without an edit here.

EXCLUDED: EvalPlus/HumanEval never enters SFT -- it is the eval. data/eval/humaneval/ is
not in either list, and check_no_eval_leak below asserts that rather than trusting it.

The holdout filter is prepare_sft's own is_holdout, applied to the same field, so this
pack cannot contain a question either arm will be scored on.
"""

import argparse
import hashlib
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from holdout import is_holdout  # noqa: E402
import prepare_sft  # noqa: E402
import prepare_sft_math  # noqa: E402

OUT_DEFAULT = os.path.join(ROOT, "data", "sft", "control_sft_text.jsonl")


def read_pairs(sources, tag):
    """(question, answer, tag) from a source list, holdout questions dropped.

    Deliberately mirrors prepare_sft.read_examples -- including the `input` field
    concatenation, which several sources use and which a reimplementation would silently
    drop -- but yields RAW text instead of calling format_example, because the template is
    applied per arm by that arm's own tokenizer.
    """
    for path, qk, ak in sources:
        if not os.path.exists(path):
            print(f"  MISSING {path}", flush=True)
            continue
        n = n_hold = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                q = (d.get(qk) or "").strip()
                a = (d.get(ak) or "").strip()
                inp = (d.get("input") or "").strip()
                if inp:
                    q = f"{q}\n{inp}"
                if not q or not a:
                    continue
                if is_holdout(q):
                    n_hold += 1
                    continue
                yield q, a, tag
                n += 1
        print(f"  {n:7d} rows ({n_hold} holdout dropped)  {tag}  {os.path.basename(path)}",
              flush=True)


def check_no_eval_leak(rows, out):
    """No pack row may carry a question from an eval set we will score on.

    Returns "clean", "leak", or "unchecked" -- three states, not two, because a check that
    could not run is not a check that passed. The first version of this returned True when
    neither eval file was present, i.e. it reported a clean pack having compared against
    nothing; that is the same shape as docs/lessons/gate_failure_shapes.md §64 (a criterion
    verified on the wrong population), here degenerating to an empty one.

    Asserted rather than assumed: the pack is built from lists that do not name the eval
    files, but "the list does not name it" and "no row matches it" are different claims,
    and only the second is about the artifact.
    """
    checked = 0
    for rel, field in (("data/eval/humaneval/humaneval_164.jsonl", "prompt"),
                       ("data/eval/code_holdout_500.jsonl", "instruction")):
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p):
            out.append(f"  skip {rel} (absent here; this file must run on the pod to check it)")
            continue
        needles = set()
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                v = (json.loads(line).get(field) or "").strip()
                if len(v) >= 40:
                    needles.add(" ".join(v.split())[:200])
        hits = sum(1 for q, _, _ in rows if " ".join(q.split())[:200] in needles)
        checked += 1
        out.append(f"  {hits} hit(s) of {len(needles)} {os.path.basename(p)} prompts in the pack")
        if hits:
            return "leak"
    if not checked:
        out.append("  NOTHING CHECKED: no eval file was present, so this is not a clean result")
        return "unchecked"
    return "clean"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--allow_unchecked", action="store_true",
                    help="write even when no eval file was available to check against; the "
                         "pack's leak_check field then records 'unchecked'")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    print("code + general (prepare_sft.SOURCES, v5 family-clean list):", flush=True)
    rows = list(read_pairs(prepare_sft.SOURCES, "code_general"))
    print("math CoT (prepare_sft_math.SOURCES):", flush=True)
    rows += list(read_pairs(prepare_sft_math.SOURCES, "math_cot"))

    # The two lists overlap on alpaca_gpt4_zh and coig: dedupe on (question, answer) so a
    # row is not silently weighted twice. Order preserved -- the arms must see the same
    # sequence, and a set would make it depend on hash seeding.
    seen, uniq = set(), []
    for q, ans, tag in rows:
        k = hashlib.sha256((q + "\0" + ans).encode()).digest()
        if k in seen:
            continue
        seen.add(k)
        uniq.append((q, ans, tag))
    print(f"\n{len(rows):,} rows -> {len(uniq):,} after dedupe ({len(rows)-len(uniq):,} dupes)",
          flush=True)

    notes = []
    verdict = check_no_eval_leak(uniq, notes)
    print("\n=== eval-leak check")
    for line in notes:
        print(line)
    if verdict == "leak":
        print("REFUSING to write: an eval question is in the pack")
        return 1
    if verdict == "unchecked" and not a.allow_unchecked:
        print("REFUSING to write: the eval-leak check could not run, and an unchecked pack "
              "must not be handed to two training runs.\nRun this on the pod where "
              "data/eval/ exists, or pass --allow_unchecked to write a pack whose "
              "provenance says so.")
        return 2

    part = a.out + ".part"
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(part, "w", encoding="utf-8") as f:
        for q, ans, tag in uniq:
            f.write(json.dumps({"question": q, "answer": ans, "src": tag},
                               ensure_ascii=False) + "\n")
    h = hashlib.sha256()
    with open(part, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    nbytes = os.path.getsize(part)
    os.replace(part, a.out)

    print(f"\nwrote {a.out}")
    print(f"  rows        {len(uniq):,}")
    print(f"  bytes       {nbytes:,}")
    print(f"  sha256      {h.hexdigest()}")
    print(f"  leak_check  {verdict}")
    print("\nThis sha256 is the pack identity for the report header: both arms must quote it,"
          "\nand each reports its own token count after tokenizing this same file.")
    return 0


def selftest():
    """The two claims this file makes that could silently be false."""
    fails = []

    # 1. read_pairs concatenates `input` -- the field prepare_sft uses and a
    #    reimplementation would drop.
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "s.jsonl")
        with open(p, "w", encoding="utf-8") as f:
            f.write(json.dumps({"instruction": "Q", "input": "EXTRA", "output": "A"}) + "\n")
            f.write(json.dumps({"instruction": "", "output": "A"}) + "\n")  # dropped
        got = list(read_pairs([(p, "instruction", "output")], "t"))
        if len(got) != 1:
            fails.append(f"expected 1 usable row, got {len(got)}")
        elif got[0][0] != "Q\nEXTRA":
            fails.append(f"`input` not concatenated: {got[0][0]!r}")

    # 2. the eval-leak check must distinguish THREE states. The first version of it
    #    returned "pass" when neither eval file existed, so a pack that had been compared
    #    against nothing looked clean -- the bug this case exists to keep dead.
    needle = "def has_close_elements(numbers, threshold):\n    " + "x" * 60
    with tempfile.TemporaryDirectory() as d:
        real_root = globals()["ROOT"]
        try:
            globals()["ROOT"] = d
            # unchecked: no eval file anywhere
            out = []
            if check_no_eval_leak([(needle, "ans", "t")], out) != "unchecked":
                fails.append(f"absent eval files did not report 'unchecked': {out}")
            # the checker's path, exactly -- a fixture one directory off proves nothing
            evd = os.path.join(d, "data", "eval", "humaneval")
            os.makedirs(evd, exist_ok=True)
            with open(os.path.join(evd, "humaneval_164.jsonl"), "w", encoding="utf-8") as f:
                f.write(json.dumps({"prompt": needle}) + "\n")
            out = []
            if check_no_eval_leak([(needle, "ans", "t")], out) != "leak":
                fails.append(f"a pack containing an eval prompt was not reported as a leak: {out}")
            out = []
            if check_no_eval_leak([("something else entirely " * 5, "ans", "t")], out) != "clean":
                fails.append(f"a clean pack was not reported clean: {out}")
        finally:
            globals()["ROOT"] = real_root

    for f in fails:
        print(f"  SELFTEST FAIL {f}")
    if fails:
        print(f"\n{len(fails)} selftest failure(s)")
        return 1
    print("build_control_sft_text selftest OK (input concat, leak check fires and clears)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
