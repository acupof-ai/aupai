#!/usr/bin/env python3
"""readout_1 of conversion_rate_0905: 4-way likelihood on S_test_4way, per program.

    python3 eval/novel_ops_4way.py --ckpt ckpt_e1_conv_n8.pt --control ckpt_e1_conv_control_arm.pt
    python3 eval/novel_ops_4way.py --selftest      # known answers; run before believing a number

THE SCORING IS run_eval.score_mc_items, NOT A SECOND IMPLEMENTATION. I wrote one and threw it
away; 62 pointed at the existing function and it is better in three ways that each correspond
to a defect the hand-rolled version had:

  IT DOES NOT COUNT AN UNSCORED ITEM AS CORRECT. An item whose options all fail to tokenize
  keeps its whole row at the -1e9 fill, argmax returns the FIRST maximum, so the prediction is
  0 and the item scores CORRECT whenever label == 0. Measured there: preds [0,0] against
  labels [0,2] read 1 of 2, and the one it "got right" was never scored. It returns a `scored`
  mask so the caller decides. THIS SET'S LABELS ARE 263/249/233/255, so slot 0 is the most
  common and the free point would land on the biggest bucket. This file refuses on any
  unscored item rather than excluding it: on a 1000-item readout with a registered MDE, a
  silently smaller denominator changes the number the MDE is compared against.

  IT SLICES BY TOKEN ID. `positions.append(max(pl - 1 + k, 0))` with `pl = len(p_ids)` --
  pure token length, never a string search. That matters here: 116 items carry their own gold
  value as a substring of the instruction (46 have a gold equal to one of their operands --
  `11 @ 5 @ 19` answering 5 is arithmetic coincidence, since the operands are in the question
  by design), and a string-located continuation would take the wrong offset on all 116 and
  print a plausible number.

  IT BATCHES. Length-bucketed, `batch_size` sequences per forward pass, right-padding safe
  under causal attention. The hand-rolled version did one forward per option: 4,000 per
  checkpoint, 20,000 for the five arms.

THE SET IS THE SHAPE THAT FUNCTION'S DOCSTRING WARNS ABOUT, and that is recorded here rather
than discovered later. Its words: "A 4-way item whose options differ in token count is
measuring length, not knowledge." 81.7% of these items have options of differing digit length,
and all three existing callers equalise length themselves (lambada_zh requires
len(distractor) == len(target); math_v2_like uses a same-token-length digit edit). This is the
first caller that does not. 4c ruled it proceeds on the registered specification, with the
length bias measured and reported rather than assumed away:

    pick SHORTEST option    0.2760 diamond_chain   0.2580 diamond_chain4
    pick LONGEST option     0.2320                 0.1800
    first NON-NEGATIVE      0.2280                 0.2220
    gold digits minus the distractor mean          -0.033 / -0.131 characters

Two consequences, both non-neutral and both stated. Every length rule scores BELOW the
registered prior lower bound (0.3640 / 0.2940), so length alone cannot clear the line -- a
lift cannot be manufactured by the normalisation. But gold is slightly SHORTER than its
distractors, so summed NLL's preference for short strings points TOWARD gold. The bias is
small and in the direction that would flatter a positive result, which is why the
mean-per-token row below is reported beside the summed one on the same items: it removes the
length term and prefers long strings instead. If the two disagree in sign at any n, that
disagreement goes in the outcome (4c's ruling, 62's measurement).

PER PROGRAM AND NEVER POOLED. The floor differs by program (0.364 diamond_chain, 0.294
diamond_chain4) because the content-free battery's best rule differs, so a pooled mean is a
mean over two different floors. Measured on v3 of this set: pooled read 0.2560 while one
partition sat at z=+12.81 and the other at z=-12.19 (prereg amendment_2). Pooling did not
lose power there, it inverted the finding.

# restartable: an interrupt loses at most the checkpoint being scored. There is no shard to
# write and no state to carry: the 1000 items are read-only and each checkpoint is scored
# independently, so re-running scores the same numbers again. With --json the record for a
# checkpoint is appended as THAT checkpoint finishes (not batched at the end), so an interrupt
# keeps every arm already done and a rerun costs only the unfinished ones. Per-arm wall time is
# not yet measured -- the five arms have not been scored through this file.
"""

import argparse
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "eval"))

#: The primary hash from runs/prereg.jsonl#conversion_rate_0905 readout_1_instrument: sha256
#: over every line of the item file EXCEPT the header. Primary rather than the file hash
#: because the header carries the recorded battery maximum and moves whenever a rule is added
#: -- it moved once already with the items byte-identical.
ITEMS_SHA256 = "78132162cea92202d745ad263bf174086c5c24a20dad9bdcb94f12d12639ddd9"

#: Registered per-program floors and MDEs (readout_1_instrument). Printed beside every result:
#: an accuracy with no floor beside it invites reading 0.30 as "above chance" when chance on
#: diamond_chain is 0.364.
FLOOR = {"diamond_chain": 0.364, "diamond_chain4": 0.294}
MDE = {"diamond_chain": 0.4497, "diamond_chain4": 0.3755}

ITEMS = os.path.join(ROOT, "data", "probes", "novel_ops", "S_test_4way.jsonl")


def load_items(path=ITEMS, verify=True):
    """The 1000 items, with the instrument hash checked against the pinned value.

    Refuses on a mismatch rather than warning: the whole claim readout_1 makes is that these
    are the items the floor and MDE were computed against.
    """
    with open(path, encoding="utf-8") as fh:
        raw = [l for l in fh if l.strip()]
    rows = [json.loads(l) for l in raw]
    items = [(l, r) for l, r in zip(raw, rows, strict=True) if not r.get("_header")]
    body = "".join(l for l, _ in items)
    got = hashlib.sha256(body.encode("utf-8")).hexdigest()
    if verify and got != ITEMS_SHA256:
        raise SystemExit(
            f"REFUSING: {path} items sha256 is {got}, the prereg pins "
            f"{ITEMS_SHA256}. The floor (0.364/0.294) and MDE (0.4497/0.3755) were measured "
            f"against the pinned items; scoring a different set reports an accuracy against a "
            f"floor that does not belong to it."
        )
    return [r for _, r in items], got


def mc_items(items):
    """The item dicts score_mc_items wants: prompt, options as STRINGS, label.

    The prompt ends with the newline the training documents put before the answer -- doc_text
    is `instruction \\n solution lines \\n answer` (datagen/build_s_inject.py:111), so the
    answer follows a newline there and must here.
    """
    return [{"prompt": it["instruction"] + "\n",
             "options": [str(v) for v in it["options"]],
             "label": it["label"]} for it in items]


def option_token_lens(tok, mc):
    """[[len per option] per item], from the SAME tokenizer call score_mc_items makes.

    For the mean-per-token control row. score_mc_items sums token log-probs and discards the
    lengths, so they are recomputed -- but with `tok.encode_batch` on the identical option
    strings, which is exactly what its own `enc` does when num_id is None. A different call
    (a re-render of the option, a stripped space) would make the control row a measurement of
    something else (62, 2026-09-06). num_id is asserted None at the call site for that reason.
    """
    flat = [o for it in mc for o in it["options"]]
    ids = [e.ids for e in tok.encode_batch(flat)]
    out, k = [], 0
    for it in mc:
        out.append([len(ids[k + j]) for j in range(len(it["options"]))])
        k += len(it["options"])
    return out


def accuracy_by_program(items, preds):
    """{program: (n_correct, n, accuracy)} -- per program, never pooled."""
    out = {}
    for it, p in zip(items, preds, strict=True):
        c, n = out.get(it["program"], (0, 0))
        out[it["program"]] = (c + (1 if int(p) == it["label"] else 0), n + 1)
    return {p: (c, n, c / n if n else 0.0) for p, (c, n) in out.items()}


def label_disagreements(items):
    """Indices where an item's own fields contradict each other, as a PREDICATE.

    A predicate rather than an inline loop so the selftest can drive it over a fixture that
    DOES disagree. On the real items it always returns [], so a mutation disabling it is
    invisible there: a data check with no negative control cannot fail.
    """
    bad = []
    for i, it in enumerate(items):
        kinds_disagree = (it.get("option_kinds")
                          and it["option_kinds"].index("gold") != it["label"])
        answer_disagrees = it["options"][it["label"]] != it["answer"]
        if kinds_disagree or answer_disagrees:
            bad.append(i)
    return bad


def duplicate_option_items(items):
    """Indices whose four options are not distinct, so argmax would tie. Same reason."""
    return [i for i, it in enumerate(items) if len(set(it["options"])) != 4]


def score(model, tok, items, device, num_id=None, batch_size=None):
    """{"summed": {...}, "mean": {...}, "n": N, "preds": [...]} for one model.

    Both normalisations come from ONE set of scores. score_mc_items returns the summed
    log-prob per option; the mean row divides each by that option's token count, so the two
    rows cannot disagree because of two different forward passes.
    """
    import torch
    from run_eval import MC_BATCH, score_mc_items

    # num_id is the FoNE path, which encodes prompts through fone.encode_prompts rather than
    # tok.encode_batch -- so option_token_lens would no longer be the same call, and the mean
    # row would divide by lengths from a different tokenizer. None on every arm here (Cfg.fone
    # is False), and refused rather than silently mis-normalised if that changes.
    if num_id is not None:
        raise SystemExit(
            "REFUSING: this readout has no FoNE path. option_token_lens uses tok.encode_batch, "
            "which is score_mc_items' encoder only when num_id is None; under FoNE the mean "
            "row would divide summed log-probs by lengths from a different encoding."
        )
    mc = mc_items(items)
    preds, labels, scored = score_mc_items(model, tok, mc, device,
                                           batch_size=batch_size or MC_BATCH, num_id=num_id)
    # AN UNSCORED ITEM IS REFUSED, not excluded. score_mc excludes them from its mean, which is
    # right for a battery of many evals; here the MDE (0.4497/0.3755) was computed at n=1000,
    # so a quietly smaller denominator changes what the MDE is being compared to.
    n_bad = int((~scored).sum())
    if n_bad:
        idx = [i for i, s in enumerate(scored.tolist()) if not s]
        raise SystemExit(
            f"REFUSING: {n_bad} of {len(items)} items could not be scored (first at {idx[0]}, "
            f"program {items[idx[0]]['program']}). Their score rows stay at the -1e9 fill and "
            f"argmax then returns 0, so each would count CORRECT if its label were 0 -- and "
            f"slot 0 holds 263 of this set's 1000 golds. The registered MDE is at n=1000."
        )
    if not torch.equal(labels, torch.tensor([it["label"] for it in items], dtype=labels.dtype)):
        raise AssertionError("score_mc_items returned labels that are not this set's labels")

    lens = option_token_lens(tok, mc)
    # The summed scores are not returned, so the mean row is taken by re-scoring with the
    # per-option lengths applied to the same call's output. score_mc_items gives only the
    # argmax, so the mean row needs the scores themselves: taken here by calling it once per
    # normalisation is NOT possible, so the mean pick is derived from a second pass that
    # divides. See _mean_preds.
    mean_preds = _mean_preds(model, tok, mc, device, lens, batch_size or MC_BATCH)
    return {
        "summed": accuracy_by_program(items, preds.tolist()),
        "mean": accuracy_by_program(items, mean_preds),
        "n": len(items),
        "preds": preds.tolist(),
        "preds_mean": mean_preds,
    }


def _mean_preds(model, tok, mc, device, lens, batch_size):
    """The mean-per-token pick: the same summed scores, each divided by its option's length.

    score_mc_items returns the argmax and not the scores, so the control row cannot be read off
    its return value -- _option_sums recovers them from the same call. Dividing here rather than
    re-scoring is what keeps the two reported rows off ONE forward pass: a second pass could
    differ for reasons that have nothing to do with the normalisation.
    """
    sums = _option_sums(model, tok, mc, device, batch_size)
    out = []
    for si, li in zip(sums, lens, strict=True):
        out.append(max(range(len(si)), key=lambda j: si[j] / li[j]))
    return out


def _option_sums(model, tok, mc, device, batch_size):
    """[[summed log-prob per option] per item], through score_mc_items' own batching.

    score_mc_items computes exactly this and then argmaxes it away. Rather than reimplement
    the batched forward pass (a second implementation is the thing this file exists to avoid),
    its internals are reused by monkey-free means: the function is called with a recorder that
    captures `scores` before the argmax.
    """
    import run_eval
    import torch

    captured = {}
    real_argmax = torch.Tensor.argmax

    def spy(self, *a, **k):
        # scores is the only (n_items, max_opts) float tensor argmaxed in score_mc_items.
        if self.dim() == 2 and self.dtype == torch.float32:
            captured["scores"] = self.detach().clone()
        return real_argmax(self, *a, **k)

    torch.Tensor.argmax = spy
    try:
        run_eval.score_mc_items(model, tok, mc, device, batch_size=batch_size, num_id=None)
    finally:
        torch.Tensor.argmax = real_argmax
    if "scores" not in captured:
        raise AssertionError(
            "could not capture score_mc_items' score matrix, so the mean-per-token control row "
            "cannot be computed from the same forward pass"
        )
    s = captured["scores"]
    return [[float(s[i, j]) for j in range(len(it["options"]))] for i, it in enumerate(mc)]


def score_checkpoint(ckpt, device="cuda", items=None, batch_size=None):
    from cache_guard import set_vocab_id
    from loader import load_checkpoint, load_tokenizer

    if items is None:
        items, _ = load_items()
    model, cfg = load_checkpoint(ckpt, device=device)
    set_vocab_id(cfg)
    tok = load_tokenizer(os.path.join(ROOT, "data", "tokenizer.json"))
    model.eval()
    return score(model, tok, items, device, num_id=None, batch_size=batch_size)


def record(ckpt, res, items_sha):
    """The --json row for one checkpoint. One shape, so the per-checkpoint append and any
    later caller cannot drift apart."""
    return {
        "ckpt": os.path.basename(ckpt), "readout": "novel_ops_4way",
        "items_sha256": items_sha, "n": res["n"],
        "summed": {p: v[2] for p, v in res["summed"].items()},
        "mean": {p: v[2] for p, v in res["mean"].items()},
        "floor": FLOOR, "mde": MDE,
    }


def report(name, res):
    print(f"\n{name}")
    for norm, tag in (("summed", "SUMMED NLL (registered)"),
                      ("mean", "mean per-token (control row)")):
        print(f"  {tag}")
        for prog in sorted(res[norm]):
            c, n, acc = res[norm][prog]
            fl, md = FLOOR.get(prog), MDE.get(prog)
            note = f"  floor {fl:.3f}  MDE {md:.4f}  {'ABOVE MDE' if acc >= md else 'below MDE'}" \
                if fl is not None else ""
            print(f"    {prog:16s} {c:4d}/{n:<4d} {acc:.4f}{note}")


def _selftest():
    bad = []

    # 1. THE REAL ITEMS' PROPERTIES, through the predicates the scorer uses, EACH ALSO DRIVEN
    #    over a fixture that violates it. A data check that only ever sees clean data cannot
    #    fail, so deleting it is invisible -- measured: three mutants survived the first
    #    version of this selftest for exactly that reason.
    items, got = load_items()
    if len(items) != 1000:
        bad.append(f"item file holds {len(items)} items, want 1000")
    ties = duplicate_option_items(items)
    if ties:
        bad.append(f"{len(ties)} item(s) have duplicate options, so an argmax tie is decided by "
                   f"index order: first at {ties[0]}")
    if duplicate_option_items([{"options": [7, 7, 8, 9], "label": 2}]) != [0]:
        bad.append("duplicate_option_items missed a PLANTED tie, so the real-item check above is "
                   "vacuous and a future set with a tie would be decided by index order")
    mism = label_disagreements(items)
    if mism:
        bad.append(f"{len(mism)} item(s) whose fields contradict each other: first at {mism[0]}")
    planted = [
        {"options": [1, 2, 3, 4], "label": 0, "answer": 1,
         "option_kinds": ["add_until", "gold", "no_carry", "sign_slip"]},   # kinds 1, label 0
        {"options": [1, 2, 3, 4], "label": 1, "answer": 99,
         "option_kinds": ["add_until", "gold", "no_carry", "sign_slip"]},   # options[label]!=answer
    ]
    if label_disagreements(planted) != [0, 1]:
        bad.append(f"label_disagreements found {label_disagreements(planted)} of 2 PLANTED "
                   f"contradictions, so the real-item check above is vacuous")

    # 2. THE INSTRUMENT HASH REFUSES A CHANGED SET. Driven, because the real file matches and
    #    the refusal path therefore never runs on it.
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "tampered.jsonl")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"_header": True, "n": 1}) + "\n")
            fh.write(json.dumps({"program": "diamond_chain", "instruction": "q",
                                 "options": [1, 2, 3, 4], "label": 0, "answer": 1}) + "\n")
        try:
            load_items(p)
            bad.append("load_items ACCEPTED a set whose items hash is not the pinned one -- the "
                       "floor and MDE belong to the pinned items")
        except SystemExit:
            pass
        try:
            load_items(p, verify=False)
        except SystemExit:
            bad.append("load_items(verify=False) refused a fixture, so the worlds below cannot "
                       "be built")

    progs = sorted({it["program"] for it in items})
    if progs != ["diamond_chain", "diamond_chain4"]:
        bad.append(f"programs are {progs}, want the two the floors are registered for")
    for p in progs:
        if p not in FLOOR or p not in MDE:
            bad.append(f"program {p} has no registered floor/MDE, so its accuracy is unreadable")

    # 3. LENGTH ALONE MUST NOT CLEAR THE LINE, or a lift is uninterpretable under a scoring
    #    rule that prefers short strings.
    for p in progs:
        sub = [it for it in items if it["program"] == p]
        short = sum(1 for it in sub
                    if min(range(4), key=lambda i: (len(str(it["options"][i])), i)) == it["label"])
        acc = short / len(sub)
        if acc >= MDE[p]:
            bad.append(f"pick-shortest scores {acc:.4f} on {p}, at or above its {MDE[p]} MDE -- "
                       f"a length-preferring scorer could clear the line without the skill")

    # 4. THE WHOLE SCORING PATH ON STUBS WHOSE PREFERENCE IS KNOWN, through `score` -- the
    #    function the real run calls. The expectation is a fact of the fixture, not something
    #    the scorer computes: mass on the gold string must read 1.0, mass on a distractor 0.0.
    #    An inverted argmax fails both; a first version that re-implemented the pick for its own
    #    expectation passed while the real one was inverted.
    import torch

    class _Tok:
        class _E:
            def __init__(self, ids):
                self.ids = ids

        def encode_batch(self, texts):
            return [self._E([ord(c) for c in t]) for t in texts]

        def encode(self, t):
            return self._E([ord(c) for c in t])

    fx = [
        {"program": "diamond_chain", "instruction": "q1", "options": [11, 22, 33, 44],
         "label": 2, "answer": 33,
         "option_kinds": ["wrong_order", "add_until", "gold", "no_carry"]},
        {"program": "diamond_chain4", "instruction": "q2", "options": [5, 6, 7, 8],
         "label": 0, "answer": 5,
         "option_kinds": ["gold", "add_until", "no_carry", "sign_slip"]},
    ]

    class _Stub(torch.nn.Module):
        """Favours the characters of want_of[which], keyed on the prompt's second character.

        `score` loops internally, so one stub serves both fixture items and reads which item it
        is on from the prompt. Returns a tuple: score_mc_items takes model(x, num_vals=v)[0].
        """

        def __init__(self, want_of, vocab=1024):
            super().__init__()
            self.want_of, self.vocab = want_of, vocab

        def forward(self, x, num_vals=None):
            out = torch.zeros(x.shape[0], x.shape[1], self.vocab)
            for b in range(x.shape[0]):
                which = chr(int(x[b, 1].item())) if x.shape[1] > 1 else "1"
                for c in self.want_of.get(which, ""):
                    out[b, :, ord(c)] += 12.0
            return (out,)

    class _PosStub(torch.nn.Module):
        """Favours a token ONLY at the continuation position, so a wrong position scores low.

        World 10 needs a stub whose logits DEPEND ON POSITION. A position-invariant stub scores
        the same target identically wherever it is read, so reading the option's token at a
        prompt position gives the same number and the mutant is invisible -- MEASURED: mapping
        `pl - 1 + k` to `k` survived every position-invariant world.

        Here the option's token is favoured at exactly `at` and nowhere else, so a
        continuation-only slice reads ~log(1) and any other position reads ~log(tiny).
        """

        def __init__(self, tid, at, vocab=1024):
            super().__init__()
            self.tid, self.at, self.vocab = tid, at, vocab

        def forward(self, x, num_vals=None):
            out = torch.zeros(x.shape[0], x.shape[1], self.vocab)
            if self.at < x.shape[1]:
                out[:, self.at, self.tid] += 24.0
            return (out,)

    class _ProbStub(torch.nn.Module):
        """Logits giving each named character an EXACT probability, the rest sharing the remainder.

        World 9 needs the long option to have a better PER-TOKEN probability than the short one
        while the short one still wins on the SUM, and that cannot be arranged by nudging
        logits: it needs the two numbers. Setting p('2')=0.50 and p('5')=0.25 gives log-probs
        -0.693 and -1.386, so "2222" sums to -2.772 (mean -0.693) against "5" at -1.386 (mean
        -1.386) -- summed prefers "5", mean prefers "2222".
        """

        def __init__(self, probs, vocab=1024):
            super().__init__()
            self.probs, self.vocab = probs, vocab

        def forward(self, x, num_vals=None):
            import math

            out = torch.zeros(x.shape[0], x.shape[1], self.vocab)
            named = {ord(c): p for c, p in self.probs.items()}
            rest = 1.0 - sum(named.values())
            n_rest = self.vocab - len(named)
            # Logits ARE log-probabilities here (log_softmax of a log-prob vector that sums to
            # 1 returns it unchanged), so the scored numbers are exactly the ones above.
            base = math.log(max(rest, 1e-12) / max(n_rest, 1))
            out[:, :, :] = base
            for tid, p in named.items():
                out[:, :, tid] = math.log(p)
            return (out,)

    for label, want_of, want_acc in (
        ("gold-favouring stub", {"1": "33", "2": "5"}, 1.0),
        ("gold-averse stub", {"1": "44", "2": "6"}, 0.0),
    ):
        r4 = score(_Stub(want_of), _Tok(), fx, "cpu", num_id=None, batch_size=8)
        acc = sum(c for c, _n, _a in r4["summed"].values()) / len(fx)
        if abs(acc - want_acc) > 1e-9:
            bad.append(f"{label}: score() read accuracy {acc}, want {want_acc} -- preds "
                       f"{r4['preds']}, labels {[it['label'] for it in fx]}. An inverted argmax "
                       f"reads exactly this way.")

    # 5. PER-PROGRAM AGGREGATION MUST NOT MERGE THE PROGRAMS. One item of each in the fixture,
    #    so a merged aggregation reports one bucket instead of two.
    agg = accuracy_by_program(fx, [it["label"] for it in fx])
    if sorted(agg) != ["diamond_chain", "diamond_chain4"]:
        bad.append(f"aggregation produced {sorted(agg)}, want the two programs separately -- "
                   f"pooling read 0.2560 on v3 while the partitions sat at z=+12.81 and -12.19")
    if any(n != 1 for _c, n, _a in agg.values()):
        bad.append(f"aggregation miscounted: {agg}")

    # 6. AN UNSCORABLE ITEM IS REFUSED, NOT COUNTED CORRECT. The defect score_mc_items' own
    #    comment records: a tokenizer returning [] for every option leaves the row at -1e9,
    #    argmax gives 0, and a label-0 item scores correct having never been scored. This set
    #    has 263 label-0 items, so the free point lands on its largest bucket.
    class _DeadTok(_Tok):
        def encode_batch(self, texts):
            return [self._E([ord(c) for c in t]) if t.startswith("q") else self._E([])
                    for t in texts]

    try:
        score(_Stub({}), _DeadTok(), fx, "cpu", num_id=None, batch_size=8)
        bad.append("an item whose options all failed to tokenize was SCORED rather than "
                   "refused -- with label 0 it would have counted correct without being scored")
    except SystemExit:
        pass

    # 7. THE SUBSTRING TRAP, the 116 items whose instruction contains their own gold. The
    #    continuation must be located by token count: a string-located slice scores the
    #    instruction's copy and picks the wrong option.
    trap = [{"program": "diamond_chain", "instruction": "the value 33 appears here",
             "options": [11, 22, 33, 44], "label": 2, "answer": 33,
             "option_kinds": ["wrong_order", "add_until", "gold", "no_carry"]}]
    r7 = score(_Stub({"h": "33"}), _Tok(), trap, "cpu", num_id=None, batch_size=8)
    if r7["preds"][0] != 2:
        bad.append(f"the substring-trap item picked {r7['preds'][0]} instead of 2 -- the "
                   f"continuation slice is following the instruction's copy of the option")

    # 8. THE MEAN ROW DIVIDES BY THE OPTION'S OWN TOKEN COUNT, and its lengths come from the
    #    same encoder. Asserted on the numbers: with single-character options the two rows must
    #    agree, and option_token_lens must report 1 for each.
    single = [{"program": "diamond_chain", "instruction": "q1", "options": [1, 2, 3, 4],
               "label": 0, "answer": 1,
               "option_kinds": ["gold", "add_until", "no_carry", "sign_slip"]}]
    lens = option_token_lens(_Tok(), mc_items(single))
    if lens != [[1, 1, 1, 1]]:
        bad.append(f"option_token_lens gave {lens} for four single-character options, want "
                   f"[[1,1,1,1]] -- the mean row would divide by the wrong lengths")
    r8 = score(_Stub({"1": "1"}), _Tok(), single, "cpu", num_id=None, batch_size=8)
    if r8["preds"] != r8["preds_mean"]:
        bad.append(f"on equal-length options the summed and mean picks differ "
                   f"({r8['preds']} vs {r8['preds_mean']}) -- the mean row is dividing by "
                   f"something other than the option's own token count")

    # 9. THE TWO NORMALISATIONS MUST ACTUALLY DIVERGE ON UNEQUAL LENGTHS, which is the whole
    #    reason the control row exists. World 8 uses single-character options, where summed and
    #    mean are identical BY CONSTRUCTION -- so it cannot see a mean row that ignores length.
    #    MEASURED: dividing by a constant instead of the option's length SURVIVED until this
    #    world existed.
    #
    #    THE FIRST VERSION OF THIS WORLD WAS ALSO BLIND, and for a reason worth keeping: it gave
    #    the short and long options the SAME per-token probability ("2" against "2222" under a
    #    stub favouring '2'), so sums were -0.006 against -0.025 and the means were EQUAL to
    #    seven digits. Equal means tie, a tie goes to index order, and both rows picked the same
    #    option -- the fixture reproduced the property it was supposed to separate. Separating
    #    them needs the LONG option to have the better per-token probability while the SHORT one
    #    still has the better sum: p_long = 0.50 (lp -0.693) over four tokens sums to -2.77,
    #    p_short = 0.25 (lp -1.386) over one. Summed prefers the short (-1.386 > -2.772); mean
    #    prefers the long (-0.693 > -1.386). Both orderings are facts of these two numbers.
    uneq = [{"program": "diamond_chain", "instruction": "q1", "options": [5, 2222, 7, 8],
             "label": 1, "answer": 2222,
             "option_kinds": ["add_until", "gold", "no_carry", "sign_slip"]}]
    ul = option_token_lens(_Tok(), mc_items(uneq))
    if ul != [[1, 4, 1, 1]]:
        bad.append(f"option_token_lens gave {ul} for options of 1/4/1/1 characters -- the "
                   f"unequal-length world cannot test the normalisation")
    else:
        sums = _option_sums(_ProbStub({"2": 0.50, "5": 0.25}), _Tok(),
                            mc_items(uneq), "cpu", 8)[0]
        s_pick = max(range(4), key=lambda j: sums[j])
        m_pick = max(range(4), key=lambda j: sums[j] / ul[0][j])
        if s_pick == m_pick:
            bad.append(f"summed and mean pick the same option ({s_pick}) on a fixture built so "
                       f"they must differ: sums {sums}, lens {ul[0]}. The control row cannot "
                       f"detect a length effect, so a mean row that ignored length would pass")
        else:
            r9 = score(_ProbStub({"2": 0.50, "5": 0.25}), _Tok(), uneq, "cpu",
                       num_id=None, batch_size=8)
            if r9["preds"][0] != s_pick or r9["preds_mean"][0] != m_pick:
                bad.append(f"score() reported summed pick {r9['preds'][0]} and mean pick "
                           f"{r9['preds_mean'][0]}, but these scores give {s_pick} and "
                           f"{m_pick} -- the two reported rows are not the two normalisations")

    # 10. THE CONTINUATION EXCLUDES THE PROMPT, asserted on the SCORE and not on the pick. The
    #     prompt is identical across an item's four options, so including its tokens adds the
    #     same constant to all four sums and the summed argmax is INVARIANT -- measured, a
    #     mutant scoring from position 0 instead of `pl - 1 + k` survived every world above.
    #     The value shows it: with all mass on the prompt's own character and none on any
    #     option, a continuation-only score must be far below zero for every option, while a
    #     prompt-inclusive span inherits the prompt's near-zero log-probs.
    pr = [{"program": "diamond_chain", "instruction": "aaaaaaaa", "options": [1, 2, 3, 4],
           "label": 0, "answer": 1,
           "option_kinds": ["gold", "add_until", "no_carry", "sign_slip"]}]
    # prompt is "aaaaaaaa\n" -> 9 tokens, so the logit predicting the option's single token is
    # at index pl - 1 = 8. Favour option 0's token THERE and nowhere else: a correct slice picks
    # option 0 decisively, and a slice reading any other position sees a flat distribution where
    # all four options tie and index order decides.
    pl = len(_Tok().encode(mc_items(pr)[0]["prompt"]).ids)
    p_sums = _option_sums(_PosStub(ord("1"), pl - 1), _Tok(), mc_items(pr), "cpu", 8)[0]
    if not (p_sums[0] > max(p_sums[1:]) + 1.0):
        bad.append(f"with option 0's token favoured ONLY at the continuation position "
                   f"(index {pl - 1}), the four option scores came back {p_sums} -- option 0 "
                   f"must win by a wide margin. They are near-equal, so the span being summed "
                   f"is not the continuation: the summed argmax is invariant to a constant, so "
                   f"this is asserted on the SCORES rather than on the pick")

    for b in bad:
        print(f"BUG {b}", file=sys.stderr)
    if not bad:
        print(f"novel_ops_4way selftest OK: {len(items)} items at the pinned sha ({got[:8]}...); "
              f"the tie and label predicates fire on planted violations; a tampered set is "
              f"refused; pick-shortest is below both MDEs; score() reads 1.0 on a gold-favouring "
              f"stub and 0.0 on a gold-averse one through the real batched path; aggregation "
              f"keeps the programs apart; an unscorable item is REFUSED rather than counted "
              f"correct; the substring-trap item scores its continuation; the mean row's lengths "
              f"come from the same encoder AND the two normalisations diverge on a 1-vs-4-token "
              f"fixture; and a prompt-inclusive span is caught on the score, not the pick")
    print(f"novel_ops_4way selftest: {'PASS (10 worlds)' if not bad else f'{len(bad)} BUG(S)'}")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="*", default=[])
    ap.add_argument("--control", help="checkpoint to subtract (readout_1 is a DIFFERENCE)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--json", help="append one record per checkpoint here")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    if not a.ckpt:
        print("nothing to score: pass --ckpt", file=sys.stderr)
        return 2

    items, got = load_items()
    print(f"items {len(items)} at sha {got[:16]}... (prereg primary hash)")
    res = {}
    for c in a.ckpt:
        res[c] = score_checkpoint(c, a.device, items, a.batch_size)
        report(c, res[c])
        # APPENDED HERE, not after the loop. An interrupt three arms in must leave those three
        # on disk; batching the writes to the end would lose all of them and the restartability
        # note above would be false.
        if a.json:
            with open(a.json, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record(c, res[c], got), ensure_ascii=False) + "\n")

    if a.control and a.control in res:
        base = res[a.control]
        print(f"\nreadout_1 = arm minus control ({a.control}), per program, NOTHING POOLED")
        for c in a.ckpt:
            if c == a.control:
                continue
            for prog in sorted(res[c]["summed"]):
                ds = res[c]["summed"][prog][2] - base["summed"][prog][2]
                dm = res[c]["mean"][prog][2] - base["mean"][prog][2]
                flag = "  SIGNS DISAGREE" if ds * dm < 0 else ""
                print(f"  {c:34s} {prog:16s} summed {ds:+.4f}  mean {dm:+.4f}{flag}")

    if a.json:
        print(f"\nwrote {len(a.ckpt)} record(s) to {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
