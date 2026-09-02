#!/usr/bin/env python3
# restartable: loads a model and the held-out TEXT, prints numbers, writes only --json_out.
# An interrupt costs a rerun. No card needed for --emit_ids.
"""The held-out number BOTH arms of the control comparison are scored with. One file.

    # 1. each arm's keep-set, no model, seconds:
    python3 scripts/eval_heldout.py --arm ours    --emit_ids /tmp/ids_ours.txt
    python3 scripts/eval_heldout.py --arm control --emit_ids /tmp/ids_ctrl.txt
    # 2. the shared population:
    python3 scripts/eval_heldout.py --intersect /tmp/ids_ours.txt /tmp/ids_ctrl.txt \
        --emit_ids /tmp/ids_shared.txt
    # 3. one scored pass per arm, on that population:
    python3 scripts/eval_heldout.py --arm ours --ckpt ckpt_sft_control_ours.pt \
        --ids /tmp/ids_shared.txt --json_out runs/heldout_ours.json
    python3 scripts/eval_heldout.py --arm control --ckpt runs/control_lr_scan/....hf \
        --ids /tmp/ids_shared.txt --json_out runs/heldout_control.json

WHY THIS REPLACES eval_heldout_ours.py AND sft_hf_control.eval_loss. Those were two
implementations of one number, and they disagreed in two ways that a report would have
presented as an architecture result:

  POPULATION. Each arm dropped the examples too long for its own seq, so "the held-out set"
  was a different set of questions per arm. Measured 2026-09-03: our arm (seq 4096) dropped
  28 of 10,641; the control (seq 1024) dropped 1,272. Comparing losses over different
  questions is gate_failure_shapes.md §64 -- a sound criterion evaluated on the wrong
  population. Here the fix is not "drop less" but "score both arms on the intersection", so
  the number is defined on questions both models actually saw.

  DENOMINATOR. Both arms divided by the supervised bytes of ALL 10,641 examples while
  evaluating a subset, and the subset is not proportional: the dropped examples are the long
  ones, so the control dropped 12.0% of the questions but 50.5% of the supervised bytes.
  Its per-byte loss came out 2.02x too small, i.e. the error flattered the control. A
  denominator must cover exactly what was summed, so this file derives it from the evaluated
  set and never from the file's length.

  WEIGHTING. sft_hf_control.eval_loss averaged HF's per-batch mean loss over ROWS, so a row
  with 8 supervised tokens weighed the same as a row with 900. eval_heldout_ours.py summed
  NLL and divided by supervised tokens. Two different statistics under one name. This file
  sums NLL, and every reported figure is that one sum over an explicit divisor.

ONE EXAMPLE PER ROW, not packed. The packed .pt files cannot answer "which examples did
this number cover" -- several examples share a row and the ids are gone. Scoring one example
per row costs padding and buys attributability: the evaluated id list is emitted, hashed,
and must match across arms. Right-padding is safe here because the models are causal and the
pad labels are -100: a padded position cannot influence an earlier one, and its own
prediction is not summed.

THE UNIT IS LOSS PER SUPERVISED BYTE. The arms' tokenizers segment the same text into
different token counts, so per-token is not a shared unit. Bytes are, and only the
completions count because prompts are masked. Per-token is printed too, labelled
arm-internal.
"""

import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

TEXT_DEFAULT = os.path.join(ROOT, "data", "sft", "control_sft_text_heldout.jsonl")
#: Only for adding the two specials to the control tokenizer. The TEMPLATE is not written here
#: -- see format_pair.
IM_START, IM_END = "<|im_start|>", "<|im_end|>"


def read_text(path):
    """[(id, question, answer)] from the shared held-out jsonl, in file order."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if "id" not in d:
                raise SystemExit(
                    f"{path}:{i+1} has no `id` field. The id is what makes 'both arms scored "
                    f"the same examples' checkable; rebuild the pack with "
                    f"datagen/build_control_sft_text.py.")
            rows.append((d["id"], d["question"], d["answer"]))
    return rows


def format_pair(arm, question, answer):
    """(prompt, completion) -- ONE template for BOTH arms: loader.format_example.

    Not per-arm, and `arm` is deliberately unused for the template. The shared artifact is
    TEXT and the template is part of what is held identical; only the tokenizer differs. Both
    trainers already call this function (sft_hf_control.read_pack imports it, sft_math via
    prepare_sft), so writing a second copy here would put a third definition of the trained
    string in the tree -- and the first draft of this file did exactly that, with a trailing
    newline the trainers do not emit. That silently inflates the byte denominator and scores
    a string neither model was trained on.
    """
    from loader import format_example
    return format_example(question, answer)


def tokenize_arm(arm, rows, seq, model_dir=None, encode=None):
    """[(id, prompt_ids, completion_ids)] for the rows that FIT, plus the dropped ids.

    Dropped ids are returned, not counted: "12% were dropped" and "these ids were dropped"
    are different claims, and only the second lets the two arms agree on a population.

    `encode` overrides the tokenizer with a callable [(str)] -> [[int]]. Only the selftest
    passes it, and it exists so the drop rule -- the part that decides the population -- is
    checkable without either arm's tokenizer file present. A rule that can only be tested
    where the model lives is a rule that goes untested on the box that runs the hook.
    """
    pairs = [format_pair(arm, q, a) for _, q, a in rows]
    if encode is not None:
        p_ids, c_ids = encode([p for p, _ in pairs]), encode([c for _, c in pairs])
    elif arm == "ours":
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(os.path.join(ROOT, "data", "tokenizer.json"))
        p_ids = [tok.encode(p, add_special_tokens=False).ids for p, _ in pairs]
        c_ids = [tok.encode(c, add_special_tokens=False).ids for _, c in pairs]
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_dir)
        tok.add_special_tokens({"additional_special_tokens": [IM_START, IM_END]})
        p_ids = tok([p for p, _ in pairs], add_special_tokens=False)["input_ids"]
        c_ids = tok([c for _, c in pairs], add_special_tokens=False)["input_ids"]
    kept, dropped = [], []
    for (rid, _, _), pi, ci in zip(rows, p_ids, c_ids, strict=True):
        # seq+1 because the row is shifted by one for next-token prediction, matching how
        # both arms' packers size a row.
        if len(pi) + len(ci) > seq + 1 or not ci:
            dropped.append(rid)
            continue
        kept.append((rid, list(pi), list(ci)))
    return kept, dropped


def supervised_bytes_of(arm, rows, ids=None):
    """Bytes of the completions, over exactly `ids` (all rows if None).

    The cross-arm denominator. Restricted to the evaluated ids by construction -- the bug
    this signature exists to prevent was a denominator over the whole file while the sum
    covered half of it.
    """
    keep = None if ids is None else set(ids)
    total = n = 0
    for rid, q, a in rows:
        if keep is not None and rid not in keep:
            continue
        _, completion = format_pair(arm, q, a)
        total += len(completion.encode("utf-8"))
        n += 1
    return total, n


def ids_sha(ids):
    """A stable fingerprint of a population. Sorted: the arms need not agree on order."""
    return hashlib.sha256(",".join(str(i) for i in sorted(ids)).encode()).hexdigest()[:16]


def load_ours(ckpt, device):
    import torch

    # HybridLM, not "Transformer": the first version of this imported a class name that does
    # not exist in train.py -- carried over from the file this replaces without being run.
    # ImportError at load time, i.e. AFTER the arm had trained. Caught by loading the PRE-SFT
    # checkpoint on CPU before the arms finished; the selftest cannot cover this because it
    # would need a real checkpoint, so the load is probed against an existing one instead.
    # sft_math.py:25-38 is the authority on how our model is constructed.
    from train import Cfg, HybridLM

    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    for k, v in (ck.get("cfg") or {}).items():
        if hasattr(Cfg, k):
            setattr(Cfg, k, v)
    model = HybridLM(Cfg).to(device)
    sd = ck.get("model") or ck.get("state_dict") or ck
    missing, unexpected = model.load_state_dict(
        {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in sd.items()},
        strict=False)
    if missing:
        raise SystemExit(f"REFUSING: {len(missing)} parameter(s) missing from {ckpt}, e.g. "
                         f"{missing[:3]} -- a partially loaded model's loss is not this "
                         f"model's loss")
    if unexpected:
        print(f"note: {len(unexpected)} unexpected key(s) ignored, e.g. {unexpected[:3]}")
    model.eval()
    return model, ck.get("vocab_id")


def load_control(path, device):
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to(device)
    model.eval()
    return model, None


def score(model, arm, kept, device, batch, pad_id):
    """(total NLL, supervised tokens) summed over `kept`. Sum, not mean -- see module doc."""
    import torch

    # Length-sorted so a batch pads to its own longest row, not the file's. Purely a speed
    # choice: the sum is order-independent, and the ids are carried through.
    order = sorted(range(len(kept)), key=lambda i: len(kept[i][1]) + len(kept[i][2]))
    tot_loss, tot_tok = 0.0, 0
    with torch.no_grad():
        for lo in range(0, len(order), batch):
            chunk = [kept[i] for i in order[lo:lo + batch]]
            width = max(len(p) + len(c) for _, p, c in chunk)
            xs, ys = [], []
            for _, p, c in chunk:
                row = p + c
                lab = [-100] * len(p) + list(c)
                pad = width - len(row)
                xs.append(row + [pad_id] * pad)
                ys.append(lab + [-100] * pad)
            x = torch.tensor(xs, dtype=torch.long, device=device)
            y = torch.tensor(ys, dtype=torch.long, device=device)
            logits = model(x[:, :-1])
            if not torch.is_tensor(logits):
                logits = logits.logits
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(), y[:, 1:].reshape(-1),
                ignore_index=-100, reduction="sum")
            tot_loss += loss.item()
            tot_tok += int((y[:, 1:] != -100).sum())
    return tot_loss, tot_tok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm", choices=("ours", "control"))
    ap.add_argument("--ckpt", help="our .pt, or the control's saved .hf directory")
    ap.add_argument("--model_dir", default=os.path.join(ROOT, "data", "controls",
                                                        "pythia-160m-step2000"),
                    help="the control arm's BASE dir, for its tokenizer")
    ap.add_argument("--text", default=TEXT_DEFAULT)
    ap.add_argument("--seq", type=int, default=None,
                    help="the arm's row width. Default 4096 for ours, 1024 for the control -- "
                         "pass it explicitly when an arm was trained at another width, or the "
                         "keep-set will not be the one the model was trained under")
    ap.add_argument("--ids", default=None,
                    help="score only these ids (one per line). This is how both arms are held "
                         "to one population; without it each arm scores its own keep-set and "
                         "the two numbers are not comparable")
    ap.add_argument("--emit_ids", default=None,
                    help="write this arm's keep-set (or --intersect's result) and exit. "
                         "No model loaded")
    ap.add_argument("--intersect", nargs="+", default=None,
                    help="intersect these id files instead of tokenizing. Use with --emit_ids")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--json_out", default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    if a.intersect:
        if not a.emit_ids:
            print("CANNOT RUN: --intersect needs --emit_ids to write to")
            return 2
        sets = []
        for p in a.intersect:
            if not os.path.exists(p):
                print(f"CANNOT RUN: {p} does not exist")
                return 2
            with open(p, encoding="utf-8") as f:
                sets.append({int(x) for x in f.read().split() if x.strip()})
        shared = set.intersection(*sets)
        if not shared:
            print("REFUSING: the arms' keep-sets do not intersect, so there is no population "
                  "to compare them on.")
            return 1
        with open(a.emit_ids, "w", encoding="utf-8") as f:
            for i in sorted(shared):
                f.write(f"{i}\n")
        for p, s in zip(a.intersect, sets, strict=True):
            print(f"  {len(s):,} ids  {p}  ({len(s) - len(shared):,} not shared)")
        print(f"\nshared population {len(shared):,} ids  sha256 {ids_sha(shared)}")
        print(f"wrote {a.emit_ids}")
        print("\nBoth arms must be scored with --ids this file, and both must report this "
              "same sha256.")
        return 0

    if not a.arm:
        ap.error("need --arm ours|control (or --intersect, or --selftest)")
    if not os.path.exists(a.text):
        print(f"CANNOT RUN: {a.text} does not exist")
        return 2
    seq = a.seq if a.seq is not None else (4096 if a.arm == "ours" else 1024)
    rows = read_text(a.text)
    kept, dropped = tokenize_arm(a.arm, rows, seq, a.model_dir)
    print(f"arm {a.arm}  seq {seq}  text {os.path.relpath(a.text, ROOT)}")
    print(f"  {len(rows):,} examples -> {len(kept):,} fit, {len(dropped):,} dropped "
          f"({100.0*len(dropped)/max(len(rows),1):.2f}%)")
    if not kept:
        print("REFUSING: nothing fits, so there is nothing to score")
        return 1

    if a.emit_ids:
        with open(a.emit_ids, "w", encoding="utf-8") as f:
            for rid, _, _ in kept:
                f.write(f"{rid}\n")
        print(f"wrote {a.emit_ids}  ({len(kept):,} ids, sha256 "
              f"{ids_sha([r[0] for r in kept])})")
        return 0

    if a.ids:
        if not os.path.exists(a.ids):
            print(f"CANNOT RUN: {a.ids} does not exist")
            return 2
        with open(a.ids, encoding="utf-8") as f:
            want = {int(x) for x in f.read().split() if x.strip()}
        have = {rid for rid, _, _ in kept}
        absent = want - have
        if absent:
            # Not a warning: scoring 9,000 of the 9,369 ids the other arm scored is the same
            # different-population defect this file exists to close, one order smaller.
            print(f"REFUSING: {len(absent)} of the {len(want)} requested ids do not fit this "
                  f"arm at seq {seq}, e.g. {sorted(absent)[:5]}. Rebuild the shared id list "
                  f"with --intersect over BOTH arms' keep-sets at the seq each was TRAINED "
                  f"at.")
            return 1
        kept = [r for r in kept if r[0] in want]
        print(f"  restricted to --ids: {len(kept):,} examples")
    else:
        print("  NOTE: no --ids, so this is this arm's own keep-set. The number is NOT "
              "comparable to the other arm until both are scored on the intersection.")

    if not a.ckpt:
        ap.error("need --ckpt to score (or --emit_ids to only write the keep-set)")
    if not os.path.exists(a.ckpt):
        print(f"CANNOT RUN: {a.ckpt} does not exist")
        return 2

    eval_ids = [rid for rid, _, _ in kept]
    sbytes, n_ex = supervised_bytes_of(a.arm, rows, eval_ids)
    if a.arm == "ours":
        model, _ = load_ours(a.ckpt, a.device)
        from tokenizers import Tokenizer
        tk = Tokenizer.from_file(os.path.join(ROOT, "data", "tokenizer.json"))
        pad_id = tk.token_to_id("<eos>") or 0
    else:
        model, _ = load_control(a.ckpt, a.device)
        from transformers import AutoTokenizer
        pad_id = AutoTokenizer.from_pretrained(a.model_dir).eos_token_id or 0

    tot_loss, tot_tok = score(model, a.arm, kept, a.device, a.batch, pad_id)
    if tot_tok == 0:
        print("CANNOT CHECK: no supervised token was scored")
        return 2

    out = {
        "arm": a.arm, "ckpt": os.path.basename(a.ckpt.rstrip("/")), "seq": seq,
        "text": os.path.relpath(a.text, ROOT),
        "examples_in_text": len(rows), "examples_scored": n_ex,
        "dropped_overlong": len(dropped),
        "evaluated_ids_sha256": ids_sha(eval_ids),
        "restricted_to_ids": os.path.basename(a.ids) if a.ids else None,
        "total_nll": tot_loss,
        "supervised_tokens": tot_tok,
        # Over the SCORED ids only. The defect this replaces divided by the whole file.
        "supervised_bytes": sbytes,
        "nll_per_supervised_byte": tot_loss / sbytes,
        "nll_per_supervised_token": tot_loss / tot_tok,
        "note": "nll_per_supervised_byte is the cross-arm number, and only when both arms "
                "report the same evaluated_ids_sha256. per_supervised_token is arm-internal: "
                "the tokenizers segment the same text into different counts.",
    }
    print(f"  scored examples          {n_ex:,}")
    print(f"  evaluated ids sha256     {out['evaluated_ids_sha256']}   <- must match the "
          f"other arm")
    print(f"  supervised tokens        {tot_tok:,}")
    print(f"  supervised bytes         {sbytes:,}")
    print(f"  total NLL                {tot_loss:,.1f}")
    print(f"  NLL / supervised BYTE    {out['nll_per_supervised_byte']:.6f}   <- compare "
          f"across arms")
    print(f"  NLL / supervised token   {out['nll_per_supervised_token']:.6f}   (arm-internal)")
    if a.json_out:
        with open(a.json_out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"wrote {a.json_out}")
    return 0


def selftest():
    """The three defects this file replaces, each with a case that fails if it returns."""
    import json as _json
    import tempfile

    fails = []

    # 1. THE DENOMINATOR COVERS THE SCORED SET, NOT THE FILE. The replaced code divided the
    #    NLL of 9,369 examples by the bytes of 10,641. Built so the two differ by construction:
    #    the long row is dropped by a small seq, and it carries most of the bytes.
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "h.jsonl")
        rows_raw = [{"id": 1, "question": "q", "answer": "a"},
                    {"id": 2, "question": "q", "answer": "b" * 4000}]
        with open(p, "w", encoding="utf-8") as f:
            for r in rows_raw:
                f.write(_json.dumps(r) + "\n")
        rows = read_text(p)
        all_bytes, all_n = supervised_bytes_of("control", rows)
        one_bytes, one_n = supervised_bytes_of("control", rows, [1])
        if all_n != 2 or one_n != 1:
            fails.append(f"supervised_bytes_of counted {all_n}/{one_n}, expected 2/1")
        if not one_bytes < all_bytes / 10:
            fails.append(f"the restricted denominator is not much smaller: {one_bytes} vs "
                         f"{all_bytes} -- this case cannot detect a whole-file denominator")
        # and the drop that motivates it: the long row must not fit a small seq. A stand-in
        # encoder (1 id per character) so the RULE is tested, not a tokenizer's output --
        # neither arm's tokenizer file need exist for the drop rule to be checkable.
        chars = lambda ss: [list(range(len(s))) for s in ss]  # noqa: E731
        kept, dropped = tokenize_arm("control", rows, 64, encode=chars)
        if [r[0] for r in kept] != [1] or dropped != [2]:
            fails.append(f"seq 64 kept {[r[0] for r in kept]} dropped {dropped}, expected "
                         f"[1] and [2]")
        # boundary: an example of exactly seq+1 ids FITS, one of seq+2 does not. Off by one
        # here silently moves the population, and both arms would move differently.
        edge = [(9, "q", "a")]
        p_len = len(format_pair("control", "q", "a")[0])
        c_len = len(format_pair("control", "q", "a")[1])
        exact = p_len + c_len - 1                       # seq such that need == seq+1
        if tokenize_arm("control", edge, exact, encode=chars)[1]:
            fails.append(f"an example needing exactly seq+1={exact+1} ids was dropped")
        if not tokenize_arm("control", edge, exact - 1, encode=chars)[1]:
            fails.append("an example needing seq+2 ids was kept")
        # An EMPTY answer is still KEPT, and that is correct: format_example appends the stop
        # token, so the completion is "<|im_end|>" -- one supervised token, the one the model
        # must emit to stop. My first draft asserted it was dropped, which was an assertion
        # about a string format_example cannot produce. The `not ci` guard below stays as a
        # guard against a template that stops appending the stop token; it is not reachable
        # through today's template, so it is not claimed as tested.
        kept_e, drop_e = tokenize_arm("control", [(7, "q", "")], 4096, encode=chars)
        if drop_e or len(kept_e) != 1 or not kept_e[0][2]:
            fails.append(f"an empty answer did not keep its stop token: kept={kept_e} "
                         f"dropped={drop_e}")
        if format_pair("control", "q", "")[1] != IM_END:
            fails.append(f"the empty-answer completion is not the bare stop token: "
                         f"{format_pair('control', 'q', '')[1]!r} -- the case above no longer "
                         f"covers what it claims")

    # 2. A MISSING id IS A REFUSAL, not a silent smaller population. Checked through main()
    #    because the refusal lives there and a unit-level check of the set arithmetic would
    #    pass while main() still scored a subset.
    #    --arm ours, because main() builds a real tokenizer and only ours (data/tokenizer.json)
    #    is present on a dev box. The refusal being checked is arm-independent.
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "h.jsonl")
        with open(p, "w", encoding="utf-8") as f:
            f.write(_json.dumps({"id": 1, "question": "q", "answer": "a"}) + "\n")
            f.write(_json.dumps({"id": 2, "question": "q", "answer": "b" * 4000}) + "\n")
        idf = os.path.join(d, "ids.txt")
        with open(idf, "w", encoding="utf-8") as f:
            f.write("1\n2\n")   # id 2 cannot fit seq 64
        argv = sys.argv
        try:
            sys.argv = ["x", "--arm", "ours", "--text", p, "--seq", "64",
                        "--ids", idf, "--ckpt", d]
            # Wrapped: a mutant that skips the refusal falls through to loading `d` as a
            # checkpoint and raises. That would abort the selftest with a traceback -- a
            # failure, but an unlabelled one. Recorded as a named FAIL instead, so the case
            # reports the defect rather than the crash it happens to cause downstream.
            try:
                rc = main()
            except Exception as e:  # noqa: BLE001 -- any escape past the refusal is the defect
                rc = f"raised {type(e).__name__}: {e}"
            if rc != 1:
                fails.append(f"an --ids list with an unfittable id gave {rc}, expected a "
                             f"refusal (1) BEFORE any model is loaded")
            # and the same call WITHOUT --ids must not refuse: the refusal must be about the
            # id list, not about anything else in this fixture.
            sys.argv = ["x", "--arm", "ours", "--text", p, "--seq", "64",
                        "--emit_ids", os.path.join(d, "out.txt")]
            if main() != 0:
                fails.append("the fixture refuses even without --ids, so case 2 does not "
                             "isolate the id-list refusal")
        finally:
            sys.argv = argv

    # 3. BOTH TRAINERS' TEMPLATE IS THIS ONE FUNCTION. sft_hf_control.read_pack and our
    #    prepare_sft both call loader.format_example; the denominator must be bytes of the
    #    string that was actually trained on, so a local copy here is the defect (the first
    #    draft had one, with a trailing newline neither trainer emits).
    try:
        import sft_hf_control
        from loader import format_example
        if format_pair("control", "Q", "A") != format_example("Q", "A"):
            fails.append("format_pair does not return loader.format_example's pair")
        src = os.path.join(HERE, "sft_hf_control.py")
        with open(src, encoding="utf-8") as f:
            body = f.read()
        if "format_example" not in body:
            fails.append("sft_hf_control.py no longer uses loader.format_example -- the "
                         "control arm's trained string has drifted from this denominator")
        if getattr(sft_hf_control, "format_example", None) is not format_example:
            fails.append("sft_hf_control.format_example is not loader's -- two templates")
    except ImportError as e:
        fails.append(f"could not import sft_hf_control / loader: {e}")

    # 4. score() SUMS OVER TOKENS. The replaced control code averaged HF's per-batch mean over
    #    ROWS, so a row with 8 supervised tokens weighed as much as one with 900. The fixture
    #    must make those two statistics DIFFER, which needs per-token losses that are not all
    #    equal -- with uniform logits a row-mean and a token-sum agree and the case is blind.
    import torch

    class Ramp(torch.nn.Module):
        """Logits whose confidence depends on POSITION, so per-token loss varies within a row.

        With a constant-logit model every token carries the same loss and 'mean over rows' and
        'sum over tokens' coincide up to a factor -- my first fixture did that and a mutation
        that averaged per row survived it.
        """

        def forward(self, x):
            v = 8
            out = torch.zeros(x.shape[0], x.shape[1], v)
            # position t puts weight t+1 on class 0: later tokens are predicted far better,
            # so a long row's mean loss is much lower than a short row's.
            for t in range(x.shape[1]):
                out[:, t, 0] = 3.0 * (t + 1)
            return out

    m = Ramp()
    short = (1, [1, 2], [3])
    long = (2, [1, 2], [3, 4, 5, 6])

    # 4a. padding-invariance: batching a short row with a long one must not change the sum.
    a1, t1 = score(m, "ours", [short, long], "cpu", 1, 0)     # each row alone, no padding
    a2, t2 = score(m, "ours", [short, long], "cpu", 2, 0)     # batched, short row padded
    if t1 != t2 or abs(a1 - a2) > 1e-4:
        fails.append(f"padding changed the sum: {a1:.6f}/{t1} vs {a2:.6f}/{t2}")

    # 4b. THE WEIGHTING. Score the short row alone, and the long row alone. A token-weighted
    #     SUM over both equals the sum of the two. A row-weighted MEAN does not -- it is
    #     bounded by the two rows' means, and here the long row's mean is much lower.
    s_short, n_short = score(m, "ours", [short], "cpu", 1, 0)
    s_long, n_long = score(m, "ours", [long], "cpu", 1, 0)
    if abs(a1 - (s_short + s_long)) > 1e-4 or t1 != n_short + n_long:
        fails.append(f"score() is not additive over rows: {a1:.6f} vs "
                     f"{s_short:.6f}+{s_long:.6f}, tokens {t1} vs {n_short}+{n_long}")
    # The fixture is only able to detect row-weighting if the two rows' MEAN losses differ.
    # Asserted, so a future edit that flattens the logits fails here instead of going blind.
    mean_short, mean_long = s_short / max(n_short, 1), s_long / max(n_long, 1)
    if abs(mean_short - mean_long) < 0.5:
        fails.append(f"fixture is blind to row-vs-token weighting: per-token means are "
                     f"{mean_short:.4f} and {mean_long:.4f}; they must differ")
    # A row-mean-over-rows statistic would land here; the real one must not.
    row_mean = (mean_short + mean_long) / 2
    if abs(a1 / max(t1, 1) - row_mean) < 1e-6:
        fails.append(f"score()/tokens equals the mean-over-rows {row_mean:.6f} -- it is "
                     f"row-weighted, not token-weighted")
    # and the same row twice must double the sum, which no mean would do
    a4, t4 = score(m, "ours", [long, long], "cpu", 2, 0)
    if abs(a4 - 2 * s_long) > 1e-4 or t4 != 2 * n_long:
        fails.append(f"the same row twice did not double the sum ({s_long:.6f} -> {a4:.6f}): "
                     f"score() is not summing")

    # 4c. THE HF OUTPUT SHAPE. A transformers model returns a ModelOutput, not a tensor, and
    #     score() unwraps .logits. That branch is only taken by the control arm, so it would
    #     first execute at result time -- after two hours of training, with nothing to show.
    #     Exercised here with a stand-in that returns an object rather than a tensor: the sum
    #     must equal the tensor-returning model's exactly.
    class Wrapped(torch.nn.Module):
        """Same logits as Ramp, returned the way transformers returns them."""

        class Out:
            def __init__(self, logits):
                self.logits = logits

        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            return self.Out(self.inner(x))

    try:
        w_loss, w_tok = score(Wrapped(m), "control", [short, long], "cpu", 2, 0)
    except Exception as e:  # noqa: BLE001 -- an unwrap that stops working raises here
        w_loss, w_tok = None, None
        fails.append(f"an HF-shaped output raised instead of scoring: "
                     f"{type(e).__name__}: {e} -- score() no longer unwraps .logits")
    if w_loss is not None and (abs(w_loss - a2) > 1e-4 or w_tok != t2):
        fails.append(f"an HF-shaped output scored differently: {w_loss:.6f}/{w_tok} vs "
                     f"{a2:.6f}/{t2} -- score() mishandles .logits unwrapping")

    # 5. THE NAMES load_ours AND load_control IMPORT MUST EXIST. No checkpoint needed: import
    #    the symbols and check they are constructible types. The first version of load_ours
    #    imported `Transformer` from train.py, which has no such class -- copied from the file
    #    this replaces without ever being executed, so it would have raised ImportError at
    #    result time, after the arm had trained. sft_math.py:25-38 is the authority.
    #
    #    SPEAKS ON SUCCESS TOO. Every other case here is silent when it passes, which is fine
    #    while they cannot be skipped. This one CAN be skipped (train.py needs torch), and a
    #    silent pass then reads identically to a skipped check -- so it prints which branch
    #    ran. That is the same defect as a criterion evaluated on an empty population: the log
    #    looks the same whether the check happened or not.
    checked_symbols = False
    try:
        import train
        for name in ("Cfg", "HybridLM"):
            if not hasattr(train, name):
                fails.append(f"train.{name} does not exist, but load_ours imports it")
        src = open(os.path.join(ROOT, "sft_math.py"), encoding="utf-8").read()
        if "HybridLM" not in src:
            fails.append("sft_math.py no longer builds HybridLM -- load_ours may be "
                         "constructing a class our arm does not train")
        checked_symbols = True
    except ImportError as e:
        # train.py imports torch and builds at module scope; on a box without it this case
        # cannot run, and says so rather than passing.
        print(f"  SELFTEST SKIP load_ours symbols: train.py not importable here ({e}) -- "
              f"this case did NOT run; run it where train.py imports")
    if checked_symbols:
        print("  checked: train.Cfg and train.HybridLM exist, sft_math.py builds HybridLM")

    for f in fails:
        print(f"  SELFTEST FAIL {f}")
    if fails:
        print(f"\n{len(fails)} selftest failure(s)")
        return 1
    print("eval_heldout selftest OK (denominator tracks the scored set, missing id refuses, "
          "templates match both trainers, score() sums and is padding-invariant)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
