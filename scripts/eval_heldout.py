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

from loader import claim_my_cards  # noqa: E402

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
    # bf16, matching eval/run_eval.py:274 (`return model.to(torch.bfloat16), cfg`) and
    # eval/domain_loss.py:286. NOT cosmetic and NOT an optimisation:
    #
    # the checkpoint's tensors are bf16 (tok.weight, blocks.0.mixer.A_log -- verified on
    # ckpt_control_ours.pt), HybridLM(Cfg) builds fp32 parameters, and load_state_dict
    # casts each loaded tensor to its parameter's dtype -- so the weights come back UP to
    # fp32 and every eval path in this repo then casts the model back down. Leaving it fp32
    # made the KDA triton kernel raise `CUDA error: misaligned address` inside its
    # autotuner. Excluded first, in this order: row width (64 fails as readily as 4096),
    # cu=None vs doc_cu_seqlens (both fail), FLA_FLASH_KDA=0 set before any import (still
    # fails), grad checkpointing (model.eval() already disables it).
    #
    # The dtype is the one thing that differed from every path that works.
    model = model.to(torch.bfloat16)
    model.eval()
    return model, ck.get("vocab_id")


def load_control(path, device):
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to(device)
    model.eval()
    return model, None


def score(model, arm, kept, device, batch, pad_id, per_item=None):
    """(total NLL, supervised tokens) summed over `kept`. Sum, not mean -- see module doc.

    Pass per_item=[] to also collect one row per item. It exists because the aggregate cannot
    answer "what does this benchmark actually test": the held-out set's own `src` field says
    99.8% code_general while only 8.3% of the answers contain code, and 84.3% are Chinese
    prose. Splitting the number by content needs per-row NLL, and this is that hook --
    computed from the SAME logits as the aggregate, never a second scoring path.
    """
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
            # THREE shapes, one per model family, and getting this wrong is a crash at
            # result time rather than a wrong number:
            #   our HybridLM     -> (logits, hidden)   model.py:534
            #   transformers     -> ModelOutput with .logits
            #   a plain module   -> a tensor
            # My first version handled only the last two, so it died on the arm it was
            # written for. The selftest's stand-in models returned a tensor and a
            # .logits object -- exactly the two cases that worked -- which is why it
            # stayed green: a fixture covering only the shapes I had thought of.
            if isinstance(logits, tuple):
                logits = logits[0]
            elif not torch.is_tensor(logits):
                logits = logits.logits
            if logits is None:
                raise SystemExit(
                    "the model returned no logits (our HybridLM returns (None, hidden) "
                    "when called with a target -- score() must call it with input only)")
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(), y[:, 1:].reshape(-1),
                ignore_index=-100, reduction="sum")
            tot_loss += loss.item()
            tot_tok += int((y[:, 1:] != -100).sum())
            if per_item is not None:
                # PER-ROW NLL, from the SAME logits the aggregate used -- not a second
                # forward pass. reduction="none" gives one value per position; masking on
                # the label and summing per row reproduces the batch total exactly, which
                # the selftest asserts. A separate scoring path here would be a second
                # implementation of one quantity, and this file already carries the scar of
                # that (the reimplemented corpus fingerprint, 8 false "SRCFP CHANGED").
                flat = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]).float(), y[:, 1:].reshape(-1),
                    ignore_index=-100, reduction="none").view(y.shape[0], -1)
                keepmask = (y[:, 1:] != -100)
                rows_nll = (flat * keepmask).sum(dim=1)
                rows_tok = keepmask.sum(dim=1)
                for (rid, _p, c), nll, ntok in zip(chunk, rows_nll.tolist(),
                                                   rows_tok.tolist()):
                    per_item.append({"id": rid, "nll": nll, "tokens": int(ntok)})
    return tot_loss, tot_tok


def control_chatml_ids(model_dir):
    """The ChatML ids this file's control tokenizer produces, by the base-dir + re-add path.

    Separate function so the scorer and the agreement check below cannot drift apart.
    """
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    tok.add_special_tokens({"additional_special_tokens": [IM_START, IM_END]})
    return {t: tok.convert_tokens_to_ids(t) for t in (IM_START, IM_END)}


def check_control_ids_agree(model_dir, ckpt_dir):
    """Refuse if the checkpoint's own tokenizer disagrees with base-dir + re-add.

    WHY THIS EXISTS. The keep-set is emitted with no checkpoint on hand (a CPU step -- see
    heldout_crossarm.sh), so it can only tokenize via base dir + add_special_tokens. The
    scorer then loads a TRAINED checkpoint that shipped its own tokenizer. Nothing made
    those two agree; `sft_hf_control.py:302` records `chatml_token_ids` in meta.json
    precisely because "an id assigned at run time is not reconstructable from the config
    afterwards", and this file never read it.

    Today they do agree (50277/50278 measured on pythia160m_lr3e-5.hf, 2026-09-03Z), so
    this changes no number. That is the point: an id collision would move every completion's
    stop symbol to a token the model never trained on and the loss would just come out
    somewhat worse -- a wrong number, not a crash. Compare, don't assume.
    """
    import json
    from transformers import AutoTokenizer
    ours = control_chatml_ids(model_dir)
    saved = AutoTokenizer.from_pretrained(ckpt_dir)
    theirs = {t: saved.convert_tokens_to_ids(t) for t in (IM_START, IM_END)}
    bad = [f"{t}: scorer {ours[t]} vs checkpoint {theirs[t]}"
           for t in (IM_START, IM_END) if ours[t] != theirs[t]]
    # meta.json is a third witness and may be absent (an older point, or a hand-copied dir):
    # absent is fine, present-and-different is not.
    meta_p = ckpt_dir.rstrip("/").removesuffix(".hf") + ".meta.json"
    if os.path.exists(meta_p):
        with open(meta_p, encoding="utf-8") as f:
            rec = json.load(f).get("chatml_token_ids") or {}
        bad += [f"{t}: scorer {ours[t]} vs meta.json {rec[t]}"
                for t in (IM_START, IM_END) if t in rec and rec[t] != ours[t]]
    return ours, bad


def vocab_size_of(arm, model_dir=None):
    """The arm's vocabulary size, for the ln(V) sentinel below. None if unavailable."""
    try:
        if arm == "ours":
            from tokenizers import Tokenizer
            return Tokenizer.from_file(os.path.join(ROOT, "data", "tokenizer.json")).get_vocab_size()
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_dir)
        tok.add_special_tokens({"additional_special_tokens": [IM_START, IM_END]})
        return len(tok)
    except Exception:  # noqa: BLE001 -- a missing tokenizer must not fail the scoring run
        return None


def alignment_sentinel(nll_per_token, vocab):
    """(line, is_alarm): compare the loss to ln(V), the no-information baseline.

    THE CHECK THAT WOULD HAVE CAUGHT A DAY OF WASTED GPU. The v1 control scan reported
    held_out_loss_before = 10.1599 nat/token for a PRETRAINED Pythia-160M. ln(50304) is
    10.8258, so that model was reading at 94% of uniform-random -- i.e. as if it had never
    been trained. The number was in every log, in every meta.json, and quoted in the audit;
    nobody divided it by ln(V). The cause was a double shift in the TRAINING loop
    (sft_hf_control.py:405), and three checkpoints were trained on the wrong objective
    before anyone noticed.

    WHY IT WENT UNNOTICED IS THE INTERESTING PART. eval_loss's docstring claimed a double
    shift pushes the loss DOWN ("the input window at position t already contains token
    t+1, so the misalignment leaks the answer"). That is FALSE, measured on the untrained
    base model over real held-out rows:

        double shift (logits[t] -> token t+2)   10.8796 nat/token  = 1.005 x ln(V)
        correct      (logits[t] -> token t+1)    2.9597 nat/token  = 0.273 x ln(V)

    There is no leak. The window does contain t+1, but the output head at position t was
    only ever trained to predict t+1, so scoring it against t+2 reads a distribution that
    is not predicting that position at all -- it degrades to no-information, not to easier.
    A double-shifted number reads LOW only when the model was TRAINED with the same wrong
    shift, because then training and scoring agree.

    So the wrong mechanism in that docstring did not merely mis-explain: it made the real
    alarm look irrelevant. 10.16 was too HIGH, and a bug believed to push losses DOWN does
    not explain a number that is too high -- so 10.16 got filed as "an untrained model is
    just bad". A wrong causal model reclassifies correct evidence as noise.

    WHAT THIS CHECK DOES **NOT** COVER, measured against ln(50279) = 10.8253:

        100.5%  untrained, scored with a double shift    10.8796   FIRES
         93.9%  v1's held_out_loss_before, the missed alarm 10.1599  FIRES
         64.6%  v1's TRAINED wrong-objective ckpt         6.9885   quiet   <- BLIND
         67.9%  the same ckpt on its own metric           7.3505   quiet   <- BLIND
         27.3%  untrained, scored correctly               2.9597   quiet
         10.1%  our arm, trained                          1.0908   quiet

    It covers the evaluation BEFORE training and nothing after it. One epoch on the wrong
    objective pulls the loss from ~100% of ln(V) down to 65%, and 65% is not distinguishable
    by any threshold from a merely weak model on hard data -- lowering the bar to 0.6 would
    refuse honest runs instead. This is a resolution limit of the metric, not a tuning
    problem. score_skip_one() is what covers the 65% row.
    """
    import math
    if not vocab or nll_per_token <= 0:
        return None, False
    lnv = math.log(vocab)
    frac = nll_per_token / lnv
    # 0.80 is a REFUSAL threshold (1e's ruling, 2026-09-03Z), not a warning line. Set below
    # the 0.94 that was actually missed and far above the 0.27 a correctly-aligned untrained
    # model reads, so the gap it must discriminate is ~3.5x wide. A number in between is
    # ambiguous and refusing is the right answer there too: no honest run lands at 0.8.
    if frac > 0.80:
        return (f"  ALARM: {nll_per_token:.4f} nat/token is {frac:.1%} of ln(V)={lnv:.4f} "
                f"(V={vocab:,}), i.e. near no-information. A trained LM on ordinary text "
                f"reads far below this. Suspect a label-alignment defect (a shift applied "
                f"twice reads ~uniform) BEFORE reading this number as a result."), True
    return (f"  sanity: {nll_per_token:.4f} nat/token = {frac:.1%} of ln(V)={lnv:.4f} "
            f"(V={vocab:,}) -- below the no-information baseline, so the alignment is "
            f"carrying signal"), False


def score_skip_one(model, arm, kept, device, batch, pad_id):
    """(total NLL, tokens) for the SKIP-ONE objective: logits[t] scored against token t+2.

    THE CHECK THAT ACTUALLY CATCHES A WRONGLY-TRAINED CHECKPOINT. The ln(V) sentinel only
    covers the pre-training evaluation; once a model has trained an epoch on the wrong
    objective its loss leaves the no-information band entirely (measured: 64.6% of ln(V),
    which no threshold can separate from a merely weak model). What does separate them is
    asking the model BOTH questions: a correctly-trained model is much better at next-token,
    and a model trained through a double shift is much better at skip-one.

    Measured on the three discarded v1 checkpoints, over 25,215 supervised held-out tokens:
        next-token (t+1)   7.2494 nat/token
        skip-one   (t+2)   2.7663 nat/token   <- 2.6x BETTER at the wrong objective
    """
    import torch

    order = sorted(range(len(kept)), key=lambda i: len(kept[i][1]) + len(kept[i][2]))
    tot, ntok = 0.0, 0
    with torch.no_grad():
        for lo in range(0, len(order), batch):
            chunk = [kept[i] for i in order[lo:lo + batch]]
            width = max(len(p) + len(c) for _, p, c in chunk)
            if width < 3:
                continue                      # needs two positions to shift by two
            xs, ys = [], []
            for _, p, c in chunk:
                row, lab = p + c, [-100] * len(p) + list(c)
                pad = width - len(row)
                xs.append(row + [pad_id] * pad)
                ys.append(lab + [-100] * pad)
            x = torch.tensor(xs, dtype=torch.long, device=device)
            y = torch.tensor(ys, dtype=torch.long, device=device)
            logits = model(x[:, :-2])
            if isinstance(logits, tuple):
                logits = logits[0]
            elif not torch.is_tensor(logits):
                logits = logits.logits
            tot += torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(), y[:, 2:].reshape(-1),
                ignore_index=-100, reduction="sum").item()
            ntok += int((y[:, 2:] != -100).sum())
    return tot, ntok


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
    # AFTER every early return above: --emit_ids, --intersect and each CANNOT RUN path do no
    # model work and take no card, so a claim there would refuse a cardless invocation that
    # is correct (de-55).
    if str(a.device).startswith("cuda"):
        claim_my_cards("eval_heldout", note=f"{a.arm} arm on {os.path.basename(a.ckpt)}")
    if a.arm == "ours":
        model, _ = load_ours(a.ckpt, a.device)
        from tokenizers import Tokenizer
        tk = Tokenizer.from_file(os.path.join(ROOT, "data", "tokenizer.json"))
        pad_id = tk.token_to_id("<eos>") or 0
    else:
        model, _ = load_control(a.ckpt, a.device)
        from transformers import AutoTokenizer
        pad_id = AutoTokenizer.from_pretrained(a.model_dir).eos_token_id or 0
        ids_used, disagree = check_control_ids_agree(a.model_dir, a.ckpt)
        if disagree:
            print("REFUSING: the ChatML ids this scorer tokenizes with are not the ids the "
                  "checkpoint was trained with, so every completion's stop symbol would be a "
                  "token the model never saw:\n  " + "\n  ".join(disagree))
            return 1
        print(f"  chatml ids agree: {ids_used}")

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
    # The ln(V) sentinel. Printed on every run, not only when it fires: a check nobody sees
    # pass is a check nobody notices missing.
    line, alarm = alignment_sentinel(out["nll_per_supervised_token"],
                                     vocab_size_of(a.arm, a.model_dir))
    if line:
        print(line)
        out["ln_vocab_sentinel"] = {"alarm": alarm, "note": line.strip()}
    if alarm:
        # REFUSE, not warn (1e's ruling 2026-09-03Z). A printed warning is what the v1 scan
        # effectively had: 10.1599 was visible in every log and got read past. The number
        # this refusal protects is the CROSS-ARM one, so it must not be produced at all --
        # a written-down wrong number outlives the log line that qualified it.
        print("\nREFUSING to report a cross-arm number: this arm scores at or near the "
              "no-information baseline, so the label alignment is broken. Fix the alignment "
              "and re-score; do not quote nll_per_supervised_byte from this run.")
        if a.json_out:
            # Written on purpose: the refusal is evidence too, and a missing file would look
            # like the run never happened.
            out["REFUSED"] = "alignment broken -- nll_per_supervised_byte must not be quoted"
            with open(a.json_out, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2)
            print(f"wrote {a.json_out} (marked REFUSED)")
        return 1

    # THE SKIP-ONE COMPARISON. Both arms, always (1e's ruling): the arms share this scorer,
    # so exempting ours would mean the check that certifies a number is not the check the
    # other arm passed. ~50% more scoring time, three minutes -- not worth saving.
    sk_loss, sk_tok = score_skip_one(model, a.arm, kept, a.device, a.batch, pad_id)
    if sk_tok:
        sk_per_tok = sk_loss / sk_tok
        nx_per_tok = out["nll_per_supervised_token"]
        out["skip_one_nll_per_token"] = sk_per_tok
        out["next_token_nll_per_token"] = nx_per_tok
        print(f"  skip-one  (t+2)          {sk_per_tok:.6f}   (alignment check)")
        if sk_per_tok < nx_per_tok:
            ratio = nx_per_tok / max(sk_per_tok, 1e-12)
            print(f"\nREFUSING: training alignment broken (skip-one beats next-token by "
                  f"{ratio:.2f}x). This model predicts token t+2 better than t+1, which is "
                  f"what a label double shift in the TRAINING loop produces -- the model "
                  f"learned the wrong objective, so its loss is not a result to compare. "
                  f"See sft_hf_control.py's training-loop comment.")
            out["REFUSED"] = (f"training alignment broken -- skip-one beats next-token by "
                              f"{ratio:.2f}x")
            if a.json_out:
                with open(a.json_out, "w", encoding="utf-8") as f:
                    json.dump(out, f, indent=2)
                print(f"wrote {a.json_out} (marked REFUSED)")
            return 1
        print(f"  alignment OK: next-token beats skip-one by "
              f"{sk_per_tok / max(nx_per_tok, 1e-12):.2f}x")
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
    # Cases that could not run HERE, with the reason. Separate from `fails` because "did not
    # run" and "ran and passed" are different facts, and the exit line must not blur them:
    # a green summary that silently covered fewer cases is how a guard dies unnoticed.
    skips = []

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
    #
    #    THAT FILE IS GITIGNORED (.gitignore:5), so in a fresh worktree it is absent and this
    #    case used to raise "No such file or directory" out of the hook -- which reads as a
    #    broken repo, and cost two other sessions time on 2026-09-03Z before either of them
    #    suspected my selftest. The skip is LOUD and names the case, because a silent pass
    #    would be the worse failure: the guard would be gone exactly where nobody looks.
    #    Every other case runs without it -- only this one goes through main().
    tokf = os.path.join(ROOT, "data", "tokenizer.json")
    if not os.path.exists(tokf):
        skips.append("case 2 (unfittable --ids refuses): data/tokenizer.json absent "
                     "(gitignored, .gitignore:5) and this case goes through main(), which "
                     "builds a real tokenizer. Run it in a tree that has the file.")
    else:
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

    # 4d. THE TUPLE SHAPE, which is what OUR arm actually returns: HybridLM.forward gives
    #     (logits, hidden) (model.py:534). Case 4c covered a tensor and a .logits object --
    #     the two shapes I had thought of -- and score() died on the third at result time,
    #     after the card was claimed and the model loaded. A fixture built from the shapes
    #     you already handle proves only that you handle them.
    class Tupled(torch.nn.Module):
        """Same logits as Ramp, returned the way HybridLM returns them."""

        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            return self.inner(x), None

    try:
        t_loss, t_tok = score(Tupled(m), "ours", [short, long], "cpu", 2, 0)
    except Exception as e:  # noqa: BLE001
        t_loss, t_tok = None, None
        fails.append(f"a (logits, hidden) tuple raised instead of scoring: "
                     f"{type(e).__name__}: {e} -- this is OUR arm's return shape")
    if t_loss is not None and (abs(t_loss - a2) > 1e-4 or t_tok != t2):
        fails.append(f"a tuple-shaped output scored differently: {t_loss:.6f}/{t_tok} vs "
                     f"{a2:.6f}/{t2}")

    # 4e. PER-ITEM MUST RECONSTRUCT THE AGGREGATE EXACTLY. This is the known answer for the
    #     new hook: sum the per-row NLLs and you must get back the number the aggregate path
    #     produced, to floating-point equality, because both come from the same logits. If a
    #     per-item column and a published total can disagree, the per-item split of a
    #     published number is not a split of it -- it is a second measurement wearing the
    #     first one's name, which is the shape that put two statistics under `eval_loss` and
    #     eight false "SRCFP CHANGED" rows in this repo.
    items = []
    p_loss, p_tok = score(m, "ours", [short, long], "cpu", 2, 0, per_item=items)
    if abs(p_loss - a2) > 1e-9 or p_tok != t2:
        fails.append(f"passing per_item changed the aggregate: {p_loss:.9f}/{p_tok} vs "
                     f"{a2:.9f}/{t2} -- the hook must not touch the number it splits")
    if len(items) != 2:
        fails.append(f"per_item collected {len(items)} rows for 2 inputs")
    else:
        s = sum(r["nll"] for r in items)
        if abs(s - a2) > 1e-4:
            fails.append(f"per-item NLLs sum to {s:.6f}, aggregate is {a2:.6f} -- the split "
                         f"does not reconstruct the total")
        if sum(r["tokens"] for r in items) != t2:
            fails.append(f"per-item tokens sum to {sum(r['tokens'] for r in items)}, "
                         f"aggregate counted {t2}")
        if {r["id"] for r in items} != {short[0], long[0]}:
            fails.append(f"per_item ids {[r['id'] for r in items]} are not the input ids "
                         f"{[short[0], long[0]]} -- length-sorting lost the mapping")
        # EACH id MUST CARRY ITS OWN VALUE, not just the right set of values. `tokens` is the
        # witness: the fixture's two rows have 2 and 4 supervised tokens, and those are known
        # from the inputs rather than from the model. Checking only the id SET and "the NLLs
        # differ" passed a mutation that reversed the chunk before zipping -- every id present,
        # every value present, each attached to the wrong row. A per-item table that swaps rows
        # is worse than none: it names the wrong question as the one we score badly.
        want_tok = {short[0]: len(short[2]), long[0]: len(long[2])}
        got_tok = {r["id"]: r["tokens"] for r in items}
        if got_tok != want_tok:
            fails.append(f"per-item tokens are attached to the wrong ids: {got_tok} vs "
                         f"{want_tok} -- the id-to-row mapping is broken")
        # And the longer completion must carry the larger NLL here, since Ramp's logits are
        # position-independent: more supervised tokens, more summed loss.
        by_id = {r["id"]: r["nll"] for r in items}
        if by_id[long[0]] <= by_id[short[0]]:
            fails.append(f"the 4-token row scored {by_id[long[0]]:.6f} <= the 2-token row's "
                         f"{by_id[short[0]]:.6f} under position-independent logits")
        # The rows must differ: a hook that returned the batch mean for every row would pass
        # every check above.
        if len({round(r["nll"], 6) for r in items}) == 1:
            fails.append("both per-item NLLs are identical -- the hook is reporting a batch "
                         "aggregate per row, not a per-row value")
        # And per_item=None must stay the default path, or every existing caller changes
        # behaviour silently.
        n_loss, _ = score(m, "ours", [short, long], "cpu", 2, 0)
        if abs(n_loss - p_loss) > 1e-9:
            fails.append("the default (per_item=None) path disagrees with the per_item path")

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
        skips.append(f"case 5 (load_ours' imported names exist): train.py not importable "
                     f"here ({e})")
    if checked_symbols:
        print("  checked: train.Cfg and train.HybridLM exist, sft_math.py builds HybridLM")

    # 6. THE MODEL MUST BE CAST TO bf16. Our checkpoints store bf16 tensors while
    #    HybridLM(Cfg) builds fp32 parameters, and load_state_dict casts loaded tensors UP
    #    to the parameter dtype -- so a load without the cast runs fp32 and the KDA triton
    #    kernel raises `CUDA error: misaligned address`. Isolated one case per process
    #    (fp32 fails sliced AND contiguous; bf16 passes both), so it is dtype, not layout.
    #
    #    No card and no checkpoint here: this reads load_ours' source for the cast, which
    #    proves the line is present, NOT that a forward pass works. Stated because the two
    #    are different claims and only a real card can make the second -- the forward is
    #    covered by actually scoring an arm, which is what found this.
    #
    #    Matched as a STATEMENT, not a substring. My first version searched load_ours' text
    #    for "to(torch.bfloat16)" and stayed green when the line was deleted, because the
    #    span also contains the phrase inside this very comment (and the split reached
    #    load_control's own cast). A check whose needle occurs in its own explanation cannot
    #    fail -- the same shape as a fixture that cannot tell two algorithms apart.
    try:
        src_lines = open(os.path.join(HERE, "eval_heldout.py"), encoding="utf-8").read()
        body = src_lines.split("\ndef load_ours", 1)[1].split("\ndef ", 1)[0]
        code = [l for l in body.splitlines() if not l.lstrip().startswith("#")]
        if not any(l.strip() == "model = model.to(torch.bfloat16)" for l in code):
            fails.append("load_ours no longer casts to bf16 -- an fp32 HybridLM raises "
                         "CUDA misaligned address in the KDA kernel")
        else:
            print("  checked: load_ours casts to bf16 (source-level; a forward pass needs "
                  "a card)")
    except (OSError, IndexError) as e:
        fails.append(f"could not read load_ours to check the bf16 cast: {e}")

    # 7. THE ChatML ID AGREEMENT CHECK MUST BITE. check_control_ids_agree compares three
    #    witnesses (this scorer's base-dir+re-add, the checkpoint's own tokenizer, and
    #    meta.json's recorded ids). Two ways to test it wrong:
    #      - assert it passes on the real checkpoint: needs a checkpoint, and it passes
    #        today, so it would prove only that today's ids happen to line up
    #      - read its source for the comparison: green when the comparison is deleted
    #    So: hand it a FABRICATED disagreement and require a non-empty `bad`. The fixture
    #    is a fake meta.json next to a fake .hf whose tokenizer is the real base dir --
    #    i.e. scorer and checkpoint agree, meta.json does not. That is the third witness,
    #    the one an "ids match the saved tokenizer" check alone would miss.
    import json as _json
    import shutil
    import tempfile
    tmp = tempfile.mkdtemp(prefix="e1_idcheck_")
    try:
        base = os.path.join(ROOT, "data", "controls", "pythia-160m-step2000")
        if not os.path.isdir(base):
            print(f"  SELFTEST SKIP chatml id agreement: {base} absent (control tokenizer "
                  f"lives on the pod) -- run --selftest there to cover case 7")
            skips.append(f"case 7 (ChatML id disagreement refuses): {os.path.relpath(base, ROOT)} "
                         f"absent -- the control tokenizer lives on the pod")
        else:
            hf = os.path.join(tmp, "fake.hf")
            os.makedirs(hf)
            for fn in os.listdir(base):
                if fn.startswith("tokenizer") or fn in ("special_tokens_map.json",
                                                        "added_tokens.json", "vocab.json",
                                                        "merges.txt", "config.json"):
                    shutil.copy(os.path.join(base, fn), hf)
            # The saved tokenizer here has NOT had the specials added, so it reports the
            # unknown-token id for both -- a real disagreement, not a synthetic one.
            _, bad_unadded = check_control_ids_agree(base, hf)
            if not bad_unadded:
                fails.append("check_control_ids_agree passed a checkpoint whose tokenizer "
                             "lacks the ChatML specials entirely -- it cannot detect an id "
                             "mismatch")
            # Now the meta.json witness, with the .hf tokenizer made to agree.
            from transformers import AutoTokenizer
            t = AutoTokenizer.from_pretrained(base)
            t.add_special_tokens({"additional_special_tokens": [IM_START, IM_END]})
            t.save_pretrained(hf)
            _, bad_clean = check_control_ids_agree(base, hf)
            if bad_clean:
                fails.append(f"check_control_ids_agree reports a disagreement on a "
                             f"checkpoint built the same way as the scorer: {bad_clean}")
            with open(os.path.join(tmp, "fake.meta.json"), "w", encoding="utf-8") as f:
                _json.dump({"chatml_token_ids": {IM_START: 999999, IM_END: 999998}}, f)
            _, bad_meta = check_control_ids_agree(base, hf)
            if not bad_meta:
                fails.append("check_control_ids_agree ignored a meta.json recording "
                             "different ChatML ids -- the recorded-id witness is dead")
            else:
                print(f"  checked: chatml id disagreement refuses "
                      f"({len(bad_unadded)} unadded, {len(bad_meta)} via meta.json)")
    except ImportError as e:
        print(f"  SELFTEST SKIP chatml id agreement: transformers unavailable ({e})")
        skips.append(f"case 7 (ChatML id disagreement refuses): transformers unavailable ({e})")

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # 8. THE ln(V) SENTINEL MUST FIRE ON THE NUMBER IT MISSED. Known-answer, using the
    #    actual v1 measurements rather than invented ones -- a fixture built from values I
    #    chose would only prove the threshold matches my choice.
    #      10.1599 nat/token at V=50,304 was in every v1 log, unquestioned (94% of ln V)
    #       2.9597 is the same untrained model scored with the CORRECT alignment (27%)
    #       0.2940 is our arm's real result per byte-adjacent token scale (well under)
    #    If the first stops alarming, the guard is gone; if either other one starts
    #    alarming, every honest run gets a false alarm and the alarm becomes noise.
    V1_MISSED, CORRECT_UNTRAINED, V = 10.1599, 2.9597, 50304
    _, a_missed = alignment_sentinel(V1_MISSED, V)
    if not a_missed:
        fails.append(f"the ln(V) sentinel does NOT fire on {V1_MISSED} nat/token at "
                     f"V={V:,} -- that is the exact number three wrong-objective "
                     f"checkpoints were trained past")
    _, a_ok = alignment_sentinel(CORRECT_UNTRAINED, V)
    if a_ok:
        fails.append(f"the ln(V) sentinel falsely alarms on {CORRECT_UNTRAINED} nat/token "
                     f"at V={V:,}, the untrained model scored CORRECTLY -- an alarm that "
                     f"fires on healthy runs will be ignored when it matters")
    _, a_trained = alignment_sentinel(1.0908, V)   # our arm's measured per-token result
    if a_trained:
        fails.append("the ln(V) sentinel alarms on our arm's real per-token result 1.0908")
    # And it must not alarm merely because a vocabulary is small: the threshold is a
    # FRACTION of ln(V), not an absolute loss. 2.0 nat/token is healthy at V=50,304 and
    # near-uniform at V=8.
    _, a_bigv = alignment_sentinel(2.0, 50304)
    _, a_smallv = alignment_sentinel(2.0, 8)
    if a_bigv or not a_smallv:
        fails.append(f"the sentinel is not scaling with V: 2.0 nat/token alarms="
                     f"{a_bigv} at V=50,304 and alarms={a_smallv} at V=8; expected "
                     f"False then True (ln 8 = 2.079)")
    if not fails:
        print("  checked: ln(V) sentinel fires on the missed 10.1599 and stays quiet on "
              "2.9597 / 1.0908, and scales with V")

    # 8b. THE ALARM MUST REACH A REFUSAL, not just a print. Case 8 checks the predicate;
    #     a predicate that returns True while main() prints it and returns 0 anyway is
    #     exactly what v1 had -- 10.1599 was visible in every log and got read past. So this
    #     asserts the refusal is WIRED: main() must return 1 and the wiring must sit after
    #     the number is computed (returning 1 before scoring would also "pass" a naive check,
    #     hence the assertion that the json still carries the loss it refused on).
    src = open(os.path.join(HERE, "eval_heldout.py"), encoding="utf-8").read()
    body = src.split("\n    line, alarm = alignment_sentinel", 1)
    if len(body) < 2:
        fails.append("could not find the sentinel call in main() -- case 8b cannot check "
                     "that the alarm is wired to a refusal")
    else:
        after = body[1].split("\n    if a.json_out:", 1)[0]
        code = "\n".join(l for l in after.splitlines() if not l.lstrip().startswith("#"))
        if "return 1" not in code:
            fails.append("the ln(V) alarm does not reach a `return 1` in main() -- it only "
                         "prints, which is what the v1 scan effectively did with 10.1599")
        elif "REFUSED" not in code:
            fails.append("the refusal does not mark the json output REFUSED -- a written "
                         "number outlives the log line that qualified it")
        else:
            print("  checked: the ln(V) alarm is wired to a refusal (return 1) that marks "
                  "its json REFUSED")

    # 8c. THE SKIP-ONE REFUSAL IS WHAT COVERS THE 65% ROW. Known answer from the real
    #     checkpoints (see score_skip_one's docstring): the discarded v1 model read 2.7663
    #     skip-one against 7.2494 next-token and MUST be refused; a healthy model reads
    #     skip-one WORSE and must pass. Checked as a decision, not by re-deriving the
    #     comparison -- a mutant that flips the operator flips this too.
    def refuses(next_tok, skip_one):
        return skip_one < next_tok
    for label, nx, sk, want in (
            ("v1 wrong-objective ckpt (must refuse)", 7.2494, 2.7663, True),
            ("untrained Pythia, correct alignment", 2.9597, 10.8796, False),
            ("our arm, trained", 1.0908, 4.0, False)):
        if refuses(nx, sk) != want:
            fails.append(f"the skip-one check {'passed' if want else 'refused'} the wrong "
                         f"way on {label}: next={nx} skip={sk}")
    # And the wiring: the comparison must reach a refusal in main(), same reason as 8b.
    sk_seg = src.split("\n    sk_loss, sk_tok = score_skip_one", 1)
    if len(sk_seg) < 2:
        fails.append("could not find the score_skip_one call in main() -- case 8c cannot "
                     "check that its comparison is wired to a refusal")
    else:
        seg = "\n".join(l for l in sk_seg[1].split("\n        print(f\"  alignment OK", 1)[0]
                        .splitlines() if not l.lstrip().startswith("#"))
        if "sk_per_tok < nx_per_tok" not in seg or "return 1" not in seg:
            fails.append("the skip-one comparison does not reach a `return 1` in main() -- "
                         "it must REFUSE, not warn")
        else:
            print("  checked: skip-one refuses the v1 wrong-objective pair (2.7663 < 7.2494) "
                  "and is wired to return 1")

    # 8d. score_skip_one MUST SCORE DIFFERENT PAIRS THAN score(), on a fixture built to be
    #     good at exactly ONE objective. Cases 8/8c check the DECISION from numbers I typed
    #     in; this one checks the two scorers actually disagree when they should, which is
    #     the part that would break if a slice were wrong.
    #
    #     MY FIRST FIXTURE HERE WAS BLIND, third time tonight for this shape. It was the
    #     Ramp from case 4 -- logits peaking at id (t % V) -- and it gave next-token and
    #     skip-one the IDENTICAL loss (25.101244 both). Not because the code was wrong: the
    #     two scorers really do score different (position, target) pairs, but the ramp's peak
    #     missed EVERY target by a constant, so both objectives read the same number. A
    #     fixture whose error is uniform across positions cannot tell two alignments apart,
    #     exactly as a constant-logit model cannot tell a row-mean from a token-sum (case 4).
    #     What works is a model built FROM THE ROW to be right for one objective only.
    #
    #     It also settles the token counts: both objectives supervise 7 tokens here, and that
    #     is correct -- shifting the label window one further drops a PROMPT -100, not a
    #     supervised token. Equal counts are not evidence of a wrong slice.
    p_ids, c_ids = [1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11, 12]
    row = p_ids + c_ids
    Vf = 17

    class _Peak(torch.nn.Module):
        """Peaks on the token `off` positions ahead, so it is good at exactly that objective."""

        def __init__(self, off):
            super().__init__()
            self.map = {t: row[t + off] for t in range(len(row) - off)}

        def forward(self, x):
            o = torch.zeros(x.shape[0], x.shape[1], Vf)
            for t in range(x.shape[1]):
                if t in self.map:
                    o[:, t, self.map[t]] = 8.0
            return o

    kept_f = [(1, p_ids, c_ids)]
    n_before = len(fails)
    for off, label, want_refuse in ((1, "good at next-token", False),
                                    (2, "good at skip-one", True)):
        m_f = _Peak(off)
        # Wrapped: dropping the extra shift makes logits and targets differ in length, so
        # score_skip_one raises a ValueError instead of returning a wrong number. Red either
        # way, but an unhandled raise aborts the selftest with a traceback that names a shape,
        # not the defect -- and prints no "SELFTEST FAIL", so grepping for that string over a
        # mutation's output shows nothing and reads as "the mutation survived".
        try:
            nx_l, nx_t = score(m_f, "ours", kept_f, "cpu", 1, 0)
            sk_l, sk_t = score_skip_one(m_f, "ours", kept_f, "cpu", 1, 0)
        except Exception as e:  # noqa: BLE001 -- any raise here is the defect
            fails.append(f"scoring a model {label} raised {type(e).__name__}: {e} -- "
                         f"score_skip_one's slices are inconsistent (logits over x[:, :-2] "
                         f"must be paired with y[:, 2:])")
            continue
        got = (sk_l / sk_t) < (nx_l / nx_t)
        if got != want_refuse:
            fails.append(f"a model {label} was {'not ' if want_refuse else ''}refused: "
                         f"next={nx_l/nx_t:.4f} skip={sk_l/sk_t:.4f} per token -- the two "
                         f"scorers are not reading different positions")
        if abs(nx_l / nx_t - sk_l / sk_t) < 1e-6:
            fails.append(f"a model {label} scores IDENTICALLY under both objectives "
                         f"({nx_l/nx_t:.6f}) -- this fixture is blind, like the Ramp that "
                         f"gave 25.101244 for both")
    # NOT for/else: that runs whenever the loop is not `break`-ed, so it printed the success
    # line even when the cases above had just appended failures. Gate on the count instead.
    if len(fails) == n_before:
        print("  checked: skip-one and next-token disagree on models built for one objective "
              "(refuse only for the skip-one model)")

    for f in fails:
        print(f"  SELFTEST FAIL {f}")
    if fails:
        print(f"\n{len(fails)} selftest failure(s)")
        return 1
    # The count is in the summary line on purpose. "selftest OK" over a run that quietly
    # covered fewer cases than it looks like is the failure mode this whole file is about.
    if skips:
        print(f"\n{len(skips)} case(s) DID NOT RUN here:")
        for s in skips:
            print(f"  - {s}")
    print("eval_heldout selftest OK (denominator tracks the scored set, missing id refuses, "
          "templates match both trainers, score() sums and is padding-invariant, per-item "
          "reconstructs the aggregate and each id carries its own row)"
          + (f" -- {len(skips)} case(s) skipped, see above" if skips else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
