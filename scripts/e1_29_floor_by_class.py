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

MEASURED BEFORE SCORING, AND IT SHRINKS THE CONFOUND THIS SCRIPT WAS BUILT TO TEST. Running the
classifier over the 10,421 scored items (pod, 2026-09-03):

    en-code    704 items ( 6.8%)   3,157,032 answer bytes (30.2%)
    en-prose   733 items ( 7.0%)   2,325,010 answer bytes (22.2%)
    zh-code    185 items ( 1.8%)     226,134 answer bytes ( 2.2%)
    zh-prose  8799 items (84.4%)   4,741,652 answer bytes (45.4%)
    ENGLISH   1437 items (13.8%)   5,482,042 bytes (52.5%)

"84.3% of the held-out items are Chinese" is true by ITEM COUNT and misleading for this question.
nll_per_supervised_byte is byte-weighted, so the published 2.004x is already 52.5% English BY WEIGHT
-- English items average 3,815 B against Chinese prose's 539 B, a 7.1x ratio, so they dominate the
denominator despite being a seventh of the items. The confound is therefore much smaller than the
item share implied, and the pre-registered readings must be read against 52.5%, not 15.7%: a large
drop on the English subset is no longer the outcome the framing predicts, because English is most of
what 2.004x already measures. An absolute item count and a byte share answer different questions,
which is the same error as reading a CJK count where a CJK fraction was needed.
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

    # BOTH ARMS ON ONE POPULATION, DECIDED BEFORE EITHER MODEL LOADS. Tokenization is CPU-only,
    # so the two kept-sets are known up front and the comparison can be made real rather than
    # merely reported.
    #
    # ON THIS DATA THIS IS A NO-OP, and the record should say so plainly. I first read
    # floor_ours.json's dropped_overlong 28 against floor_control.json's 220, saw one shared
    # supervised_bytes 10554038, and reported a two-population defect in the published pair. That
    # was WRONG. dropped_overlong is counted at eval_heldout.py:515, BEFORE the --ids restriction
    # at 531-547 -- it reports what each arm dropped from the full 10,641-row file, not from the
    # scored population. eval_heldout already refuses unless every requested id fits the arm, then
    # restricts to exactly those. Both floors carry restricted_to_ids ids_shared.txt and the same
    # evaluated_ids_sha256 cae4daf7ad59388c over 10,421 items, so the shared denominator is
    # correct and 2.004x stands.
    #
    # What survives is narrower: this script does not depend on its caller having passed the right
    # --ids. It computes the intersection itself and refuses if any class holds different items in
    # the two arms -- eval_heldout's guarantee re-checked at this layer, not replaced. The cost is
    # one CPU tokenization pass; the benefit is that a per-class ratio cannot silently become two
    # measurements side by side, which per class would be worse than a small bias because the
    # longest items are also the ones most likely to be code and English.

    tok = {}
    for arm, seq in (("ours", a.seq_ours), ("control", a.seq_ctrl)):
        kw = {} if arm == "ours" else {"model_dir": a.ctrl_dir}
        tok[arm] = E.tokenize_arm(arm, rows, seq, **kw)
        print(f"  {arm}: tokenized {len(tok[arm][0]):,}, {len(tok[arm][1])} overlong at seq {seq}")
    both = {int(r[0]) for r in tok["ours"][0]} & {int(r[0]) for r in tok["control"][0]}
    if not both:
        sys.exit("REFUSING: the two arms share no scorable item")
    out["scored_population"] = {
        "n": len(both), "sha": E.ids_sha(sorted(both)),
        "dropped_by_ours_only": len(tok["ours"][0]) - len(both),
        "dropped_by_control_only": len(tok["control"][0]) - len(both),
        "why": "both arms scored EXACTLY these ids, checked here rather than assumed from the "
               "caller's --ids. On this data it is a no-op: the published floors already restrict "
               "to ids_shared.txt and both report evaluated_ids_sha256 cae4daf7ad59388c over "
               "10,421 items, so 2.004x is sound. (I first misread floor_*.json's dropped_overlong "
               "28 vs 220 as evidence of two populations; that field is counted before the --ids "
               "restriction and describes the 10,641-row file, not the scored set.)"}

    print(f"  intersection: {len(both):,} items (sha {out['scored_population']['sha']}), "
          f"ours-only {out['scored_population']['dropped_by_ours_only']}, "
          f"control-only {out['scored_population']['dropped_by_control_only']}")

    for arm, ckpt in (("ours", a.ours_ckpt), ("control", a.ctrl_dir)):
        kept = [r for r in tok[arm][0] if int(r[0]) in both]
        dropped = tok[arm][1]
        if len(kept) != len(both):
            sys.exit(f"REFUSING: {arm} kept {len(kept)} of the {len(both)} shared ids")
        # PAD COMES FROM THE TOKENIZER, NOT FROM THE LOADER'S SECOND RETURN VALUE. load_ours
        # returns (model, ck.get("vocab_id")) -- a vocab identifier string -- and load_control
        # returns (model, None). I destructured both as `model, pad` and passed the string
        # straight into score(), which died on `torch.tensor(xs, dtype=torch.long)` with
        # "'str' object cannot be interpreted as an integer" AFTER loading the model and
        # tokenizing 10,421 items. A two-tuple's second slot is not labelled by what the caller
        # needs it to be. This is how eval_heldout's own main() derives it (lines 563-569).
        if arm == "ours":
            model, _vocab_id = E.load_ours(os.path.join(ROOT, ckpt), "cuda")
            from tokenizers import Tokenizer
            tk = Tokenizer.from_file(os.path.join(ROOT, "data", "tokenizer.json"))
            pad_id = tk.token_to_id("<eos>") or 0
        else:
            model, _ = E.load_control(ckpt, "cuda")
            from transformers import AutoTokenizer
            pad_id = AutoTokenizer.from_pretrained(ckpt).eos_token_id or 0
        if not isinstance(pad_id, int):
            sys.exit(f"REFUSING: {arm} pad_id is {type(pad_id).__name__} {pad_id!r}, not an int -- "
                     f"score() puts it straight into a long tensor")
        items = []
        tot, ntok = E.score(model, arm, kept, "cuda", a.batch, pad_id, per_item=items)

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
        tb = sum(d["bytes"] for d in agg.values())
        out["arms"][arm] = {"total_nll": tot, "tokens": ntok, "dropped": len(dropped),
                            "n_scored": len(kept), "bytes": tb,
                            "nll_per_byte": tot / tb if tb else None, "by_class": agg}
        print(f"  {arm}: {len(kept):,} scored, {tot / tb:.6f} nll/byte on the shared population")
        del model
        torch.cuda.empty_cache()

    # THE WHOLE-POPULATION GAP ON ONE POPULATION, so the per-class numbers below have a headline
    # to be a split OF. Printed next to the published 2.004x rather than replacing it: this is a
    # different (and correct) population, not a re-run of that measurement.
    ob = out["arms"]["ours"]["nll_per_byte"]
    kb = out["arms"]["control"]["nll_per_byte"]
    out["gap_shared_population"] = kb / ob
    print(f"\nSHARED-POPULATION GAP: ours {ob:.6f}  ctrl {kb:.6f}  gap {kb / ob:.3f}x")
    print("  (this should reproduce the published 2.004x: both floors already restricted to "
          "ids_shared.txt, so the intersection above is that same population)")

    # THE GAP, PER CLASS, on the shared population -- so a class's two arms hold the SAME items
    # and the ratio is a comparison rather than two measurements side by side. An n mismatch is
    # now impossible by construction, so it refuses instead of printing a caveat.
    print(f"\n{'class':<10} {'n':>7} {'ours/byte':>11} {'ctrl/byte':>11} {'gap':>7} "
          f"{'bytes':>12}")
    gaps = {}
    for c in sorted(set(out["arms"]["ours"]["by_class"]) | set(out["arms"]["control"]["by_class"])):
        o = out["arms"]["ours"]["by_class"].get(c)
        k = out["arms"]["control"]["by_class"].get(c)
        if not o or not k or o["n"] != k["n"] or o["bytes"] != k["bytes"]:
            sys.exit(f"REFUSING: class {c} is not the same items in both arms "
                     f"(ours {o and o['n']}/{o and o['bytes']}B, "
                     f"ctrl {k and k['n']}/{k and k['bytes']}B) -- the shared-population "
                     f"restriction above should have made this unreachable")
        g = k["nll_per_byte"] / o["nll_per_byte"]
        gaps[c] = {"ours_nll_per_byte": o["nll_per_byte"], "ctrl_nll_per_byte": k["nll_per_byte"],
                   "gap": g, "n": o["n"], "bytes": o["bytes"],
                   "ours_nll": o["nll"], "ctrl_nll": k["nll"]}
        print(f"{c:<10} {o['n']:>7,} {o['nll_per_byte']:>11.6f} "
              f"{k['nll_per_byte']:>11.6f} {g:>6.3f}x {o['bytes']:>12,}")
    out["gap_by_class"] = gaps

    # THE SPLIT MUST RECONSTRUCT THE HEADLINE, not merely sit beside it: byte-weighting the four
    # classes back together has to return the whole-population gap.
    for arm in ("ours", "control"):
        sb = sum(d["bytes"] for d in out["arms"][arm]["by_class"].values())
        sn = sum(d["nll"] for d in out["arms"][arm]["by_class"].values())
        if sb != out["arms"][arm]["bytes"] or abs(sn - out["arms"][arm]["total_nll"]) > 1e-3:
            sys.exit(f"REFUSING: {arm}'s classes sum to {sn:.3f} nll / {sb} bytes but the pass "
                     f"was {out['arms'][arm]['total_nll']:.3f} / {out['arms'][arm]['bytes']}")

    en = [c for c in gaps if c.startswith("en-")]
    if en:
        no = sum(gaps[c]["ours_nll"] for c in en)
        bo = sum(gaps[c]["bytes"] for c in en)
        nk = sum(gaps[c]["ctrl_nll"] for c in en)
        out["english_only"] = {"ours_nll_per_byte": no / bo, "ctrl_nll_per_byte": nk / bo,
                               "gap": nk / no, "n": sum(gaps[c]["n"] for c in en), "bytes": bo}
        zh_share = 1.0 - bo / out["arms"]["ours"]["bytes"]
        print(f"\nENGLISH ONLY ({sum(gaps[c]['n'] for c in en):,} items, {bo:,} bytes, "
              f"{100 * (1 - zh_share):.1f}% of the population): "
              f"ours {no / bo:.6f}  ctrl {nk / bo:.6f}  gap {nk / no:.3f}x")
        print(f"  against the shared-population gap {out['gap_shared_population']:.3f}x "
              f"(published, all items: 2.004x)")
        print("The English subset is what the control was trained for. Read per the pre-registered "
              "readings in this file's docstring.")
    with open(os.path.join(ROOT, a.out), "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
