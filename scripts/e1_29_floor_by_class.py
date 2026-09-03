#!/usr/bin/env python3
"""Split the 2.00x floor gap by what the held-out items actually ARE.

# restartable: two scored passes writing one JSON, ~4 min on one card. An interrupt costs the
# unfinished pass; the JSON is written once at the end and is complete or absent.

THE QUESTION, WRITTEN BEFORE THE NUMBERS EXIST. Section 5.0's floor gap is 2.00x (ours 0.450964
against the control's 0.903758) and section 4 reads it as "the gap is mostly made in pretraining".
But the control is Pythia-160M, trained on the Pile -- an ENGLISH corpus -- and 84.3% of the
held-out items are Chinese prose. So some unknown share of 2.00x is not "our model is better", it
is "our model has seen Chinese and theirs has not". That confound has never been excluded, it runs
in OUR favour, and it is measurable, which is why it gets measured rather than noted.

    On the ENGLISH items alone, how much of 2.00x survives?

A LARGE DROP AND A SMALL DROP BOTH MEAN SOMETHING, and both readings are fixed here so neither can
be chosen after the fact:
  - if the English-only gap is far below 2.00x, the published gap is substantially a language
    artifact and section 4's "made in pretraining" reading is over-claimed;
  - if it stays near 2.00x, the gap survives the confound and the reading is strengthened;
  - if it INVERTS on English, the published direction is wrong on the items the control was
    actually trained for.

WHY THE FLOORS AND NOT THE SFT POINTS. The two floor models both still exist
(ckpt_p200m_4b_0902.pt and /tmp/e1_untrained.hf, the directory behind the published 0.903758 --
runs/e1_control_model_fp.json). All five control SFT checkpoints are GONE, so the post-SFT
comparison cannot be split by anything, ever. The floors are what is left, and they are also where
the pretraining claim lives.

THE CLASSES ARE MINE, NOT THE DATA'S, and that is stated in the output. The `src` field says 99.8%
code_general; measured, 8.3% of answers contain code. A label that wrong cannot carry a
stratification, so the split is on measured content: Chinese characters in the item, and code
markers in the answer.
"""
import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

#: Code markers looked for in the ANSWER. Deliberately narrow: a brace or a semicolon appears in
#: Chinese prose about programming, so those are not markers. Under-calling code is the safe
#: direction here -- it puts ambiguous items in "prose", where they dilute a prose-vs-code
#: contrast rather than manufacture one.
CODE_MARKS = ("```", "def ", "class ", "function ", "import ", "SELECT ", "public ",
              "#include", "println", "print(", "return ")
#: A FRACTION, not a count. An absolute floor of 20 CJK characters called a 5-character Chinese
#: question "English" and a 19-character Chinese sentence "English" -- both caught by the test's
#: real-shaped fixtures before this ran on a card. What separates a Chinese item from an English
#: one that mentions a Chinese word is the SHARE of the text that is CJK, and that is
#: length-independent. 5% is low because CJK is dense: a Chinese sentence with code blocks and
#: latin identifiers still clears it, while "the character 中 means middle" does not.
ZH_MIN_FRAC = 0.05
#: Below this many total characters the fraction is noisy, so a single CJK character in a 10-char
#: string would flip the class. Such items are called English unless they are MOSTLY CJK.
SHORT_CHARS = 40


def classify(question, answer):
    t = question + answer
    zh = len(re.findall(r"[一-鿿]", t))
    frac = zh / len(t) if t else 0.0
    if len(t) < SHORT_CHARS:
        is_zh = frac >= 0.30
    else:
        is_zh = frac >= ZH_MIN_FRAC
    code = any(m in answer for m in CODE_MARKS)
    return ("zh" if is_zh else "en") + "-" + ("code" if code else "prose")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="runs/heldout_v2/ids_shared.txt")
    ap.add_argument("--text", default="data/sft/control_sft_text_heldout.jsonl")
    ap.add_argument("--ours_ckpt", default="ckpt_p200m_4b_0902.pt")
    ap.add_argument("--ctrl_dir", default="/tmp/e1_untrained.hf")
    ap.add_argument("--out", default="runs/e1_29_floor_by_class.json")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq_ours", type=int, default=4096)
    ap.add_argument("--seq_ctrl", type=int, default=2048)
    a = ap.parse_args()

    import eval_heldout as E

    rows = E.read_text(os.path.join(ROOT, a.text))
    keep = {int(x) for x in open(os.path.join(ROOT, a.ids)) if x.strip()}
    rows = [r for r in rows if int(r[0]) in keep]
    if not rows:
        sys.exit("REFUSING: no rows matched the ids file")
    cls = {int(rid): classify(q, ans) for rid, q, ans in rows}
    print(f"{len(rows):,} items on the published population "
          f"(sha {E.ids_sha([int(r[0]) for r in rows])})")

    # THE CONTROL FLOOR'S MODEL IS NAMED, NOT DEFAULTED. eval_heldout's --model_dir default is
    # data/controls/pythia-160m-step2000, which is NOT what produced the published 0.903758.
    if not os.path.isdir(a.ctrl_dir):
        sys.exit(f"REFUSING: {a.ctrl_dir} absent -- it is the model behind the published control "
                 f"floor 0.903758 and nothing in git can reconstruct it "
                 f"(runs/e1_control_model_fp.json records its weight hash)")

    import torch
    out = {"classes_defined_by": "e1, from measured content -- NOT the data's own `src` field, "
           "which labels 99.8% code_general while 8.3% of answers contain code",
           "code_marks": list(CODE_MARKS), "zh_min_frac": ZH_MIN_FRAC,
           "zh_short_item_chars": SHORT_CHARS,
           "ours_ckpt": a.ours_ckpt, "ctrl_dir": a.ctrl_dir,
           "population_sha": E.ids_sha([int(r[0]) for r in rows]), "arms": {}}

    for arm, ckpt, seq in (("ours", a.ours_ckpt, a.seq_ours), ("control", a.ctrl_dir, a.seq_ctrl)):
        if arm == "ours":
            kept, dropped = E.tokenize_arm("ours", rows, seq)
            model, pad = E.load_ours(os.path.join(ROOT, ckpt), "cuda")
        else:
            kept, dropped = E.tokenize_arm("control", rows, seq, model_dir=ckpt)
            model, pad = E.load_control(ckpt, "cuda")
        items = []
        tot, ntok = E.score(model, arm, kept, "cuda", a.batch, pad, per_item=items)
        # PER-ITEM MUST RECONSTRUCT THE PASS, here too and not only in the selftest: the split
        # below is only a split of this number if it adds back up to it.
        s = sum(r["nll"] for r in items)
        if abs(s - tot) > 1e-3:
            sys.exit(f"REFUSING: {arm} per-item NLLs sum to {s:.6f} but the pass totalled "
                     f"{tot:.6f} -- the split is not a split of this number")
        byid = {int(rid): (q, ans) for rid, q, ans in rows}
        agg = {}
        for r in items:
            c = cls[int(r["id"])]
            d = agg.setdefault(c, {"n": 0, "nll": 0.0, "bytes": 0, "tokens": 0})
            d["n"] += 1
            d["nll"] += r["nll"]
            d["tokens"] += r["tokens"]
            _q, ans = byid[int(r["id"])]
            d["bytes"] += len(E.format_pair(arm, _q, ans)[1].encode("utf-8"))
        for c, d in agg.items():
            d["nll_per_byte"] = d["nll"] / d["bytes"] if d["bytes"] else None
        out["arms"][arm] = {"total_nll": tot, "tokens": ntok, "dropped": len(dropped),
                            "by_class": agg}
        print(f"  {arm}: {len(kept):,} scored, {len(dropped)} dropped")
        del model
        torch.cuda.empty_cache()

    # THE GAP, PER CLASS. Only classes BOTH arms scored can be compared -- the control drops long
    # rows at seq 2048 that we keep at 4096, so a class where the two arms scored different items
    # is not a comparison. Reported with its own n per arm so a mismatch is visible.
    print(f"\n{'class':<10} {'n(ours)':>8} {'n(ctrl)':>8} {'ours/byte':>11} {'ctrl/byte':>11} "
          f"{'gap':>7}")
    gaps = {}
    for c in sorted(set(out["arms"]["ours"]["by_class"]) | set(out["arms"]["control"]["by_class"])):
        o = out["arms"]["ours"]["by_class"].get(c)
        k = out["arms"]["control"]["by_class"].get(c)
        if not o or not k:
            print(f"{c:<10} {'-' if not o else o['n']:>8} {'-' if not k else k['n']:>8} "
                  f"{'ONE ARM ONLY':>31}")
            continue
        g = k["nll_per_byte"] / o["nll_per_byte"]
        gaps[c] = {"ours_nll_per_byte": o["nll_per_byte"], "ctrl_nll_per_byte": k["nll_per_byte"],
                   "gap": g, "n_ours": o["n"], "n_ctrl": k["n"],
                   "bytes_ours": o["bytes"], "bytes_ctrl": k["bytes"]}
        print(f"{c:<10} {o['n']:>8,} {k['n']:>8,} {o['nll_per_byte']:>11.6f} "
              f"{k['nll_per_byte']:>11.6f} {g:>6.3f}x")
    out["gap_by_class"] = gaps

    en = [c for c in gaps if c.startswith("en-")]
    if en:
        no = sum(gaps[c]["ours_nll_per_byte"] * gaps[c]["bytes_ours"] for c in en)
        bo = sum(gaps[c]["bytes_ours"] for c in en)
        nk = sum(gaps[c]["ctrl_nll_per_byte"] * gaps[c]["bytes_ctrl"] for c in en)
        bk = sum(gaps[c]["bytes_ctrl"] for c in en)
        out["english_only"] = {"ours_nll_per_byte": no / bo, "ctrl_nll_per_byte": nk / bk,
                               "gap": (nk / bk) / (no / bo)}
        print(f"\nENGLISH ONLY: ours {no / bo:.6f}  ctrl {nk / bk:.6f}  "
              f"gap {(nk / bk) / (no / bo):.3f}x   (published, all items: 2.004x)")
        print("The English subset is the one the control was trained for. Read against 2.004x per "
              "the pre-registered readings in this file's docstring.")
    with open(os.path.join(ROOT, a.out), "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
