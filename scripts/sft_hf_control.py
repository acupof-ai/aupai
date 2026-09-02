#!/usr/bin/env python3
# restartable: writes only the output checkpoint, at the end. An interrupt costs the run.
# One card, no DDP -- 162M params at seq 1024 fits, and a single-card run has no rank-skew
# failure mode to debug on a control arm. (162,322,944 measured on the real checkpoint;
# the 212M this line used to claim counted BUFFERS as parameters and is retracted.)
"""SFT a HuggingFace causal LM on the shared control text pack. One card, ~1h.

    python3 scripts/sft_hf_control.py \
        --model data/controls/pythia-160m-step2000 \
        --pack data/sft/control_sft_text_train.jsonl \
        --out ckpt_sft_pythia160m_control.pt
    (the held-out file is found beside it: control_sft_text_heldout.jsonl)
    python3 scripts/sft_hf_control.py --selftest      # no card, no model needed

WHY THIS FILE EXISTS. sft_math.py reads our own checkpoint format and builds our own model;
Pythia-160m is a HF GPTNeoXForCausalLM. Rather than teach sft_math.py a second model
family, this reproduces the parts of it that must be IDENTICAL for the comparison to mean
anything, and states plainly the parts that cannot be.

IDENTICAL to our arm, by construction:
  the example set and its order   -- the same _train.jsonl, same sha256
  the held-out set                -- the same _heldout.jsonl, split ONCE at the text level
                                     so both arms hold out the same example ids
  the ChatML template             -- scripts.loader.format_example, imported not copied
  the loss mask                   -- prompt masked to -100, completion supervised
                                     including its <|im_end|> (the stop token must be
                                     supervised or the model never stops)
  the packing rule                -- whole examples only, never split, over-length dropped,
                                     right-padded with eos carrying no loss
  the LR schedule SHAPE           -- train.lr_mult, imported: absolute warmup then a cosine
                                     warmdown over the last cfg.warmdown of the run
  epochs                          -- default 1, the sft_math.py default

NECESSARILY DIFFERENT, and this belongs in the report header rather than a footnote:
  the tokenizer   ours vs a 50,304-entry NeoX BPE. The same TEXT yields a different token
                  count per side; both numbers get reported. A shared .pt is impossible and
                  check_sft_ready.py:check_vocab would correctly refuse one.
  the optimizer   our arm uses Muon (lr 0.01) on 2D matrices with AdamW on embeddings and
                  scalars. Muon on a foreign model is a second intervention on top of the
                  data, so this arm uses AdamW throughout. Any claim from this comparison
                  must therefore be about data and architecture TOGETHER, not either alone.
  the LR          AND IT IS NOT DERIVABLE FROM OURS, which the first version of this file got
                  wrong. It set 1e-2 = Cfg.embed_lr (0.1) x sft_math --lr_scale (0.1) and
                  called that "the same magnitude". Two errors in one: Cfg.embed_lr applies
                  to the EMBEDDING GROUP ONLY (Muon carries the 2D matrices at 0.01, and
                  Muon's update is orthogonalised, so its lr is not on the same scale as
                  AdamW's at all), and a per-group lr from our optimizer says nothing about a
                  foreign model's every-parameter lr. Measured on a CPU smoke test over 4
                  steps: 1e-2 DIVERGES (loss 11.52 -> 13.27), 1e-4 descends (-> 2.49), 2e-5
                  is too slow (-> 7.60). The default is therefore 1e-4, chosen by measurement
                  on this model, and the report must say the arms' LRs were each set for their
                  own optimizer rather than matched.
  the ChatML tokens  ours are single special tokens; the NeoX BPE has no <|im_start|>, so
                  they tokenize as several ordinary tokens. Added to the tokenizer as real
                  special tokens here so the two sides both learn a dedicated stop symbol --
                  the alternative, letting them split, would handicap the control for a
                  reason unrelated to the question. NO resize: Pythia's config.vocab_size
                  (50,304) already exceeds its tokenizer (50,277), so the two ids land on
                  rows no token ever addressed -- near-init, NOT trained (measured; see the
                  resize comment in main()).
  the tied head   OURS IS TIED (model.py:322 sets head.weight = tok.weight, one tensor);
                  Pythia's embed_in and embed_out are two separate 38,633,472-param tensors.
                  So non-embedding params (85,056,000 here) is the only figure comparable
                  across the two arms, and a total-param comparison is not.
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import eval_heldout_ours  # noqa: E402  -- the shared supervised-byte denominator
from loader import IM_END, format_example  # noqa: E402

SEQ = 1024  # Pythia-160m's max_position_embeddings is 2048; 1024 halves the step cost and
#: still holds the overwhelming majority of these examples whole. Reported, not silent:
#: the drop count from this choice is printed and lands in the report.


def lr_mult_shape(step, total, warmup, warmdown, final_frac):
    """train.lr_mult's schedule, re-expressed with explicit numbers.

    train.py imports torch and builds a model at module scope, so importing it on a
    tokenizer-only selftest is not free; the function is 8 lines and its parameters are
    named here so the two schedules can be asserted equal (see selftest).
    """
    if step < warmup:
        return (step + 1) / warmup
    wd_steps = max(1, int(warmdown * total))
    wd_start = total - wd_steps
    if step < wd_start:
        return 1.0
    progress = min(1.0, (step - wd_start) / wd_steps)
    return final_frac + (1 - final_frac) * 0.5 * (1 + math.cos(math.pi * progress))


def read_pack(path):
    """One shared text file -> ([(prompt, completion)], [example id]) through OUR template.

    The train/held-out split is NOT made here. build_control_sft_text.py writes
    <name>_train.jsonl and <name>_heldout.jsonl, and BOTH arms read BOTH files (fb's ruling
    2026-09-02). Each arm splitting for itself was the bug: our arm would have trained on the
    2% the control holds out, so it would both train on more data and see the control's
    validation set.

    The returned ids are the "id" field the builder stamps -- the example's index in the
    deduped set. They make "the two arms held out the same examples" checkable from the
    artifacts, instead of a claim that two scripts implement the same rule.

    An lr scan MUST be selected on held-out loss. The training loss of the last step rewards
    whichever lr memorised the most, which is precisely the artefact the scan exists to rule
    out ("we won because the control was undertuned" would just become "the control's lr was
    picked to overfit").
    """
    pairs, ids = [], []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            pairs.append(format_example(d["question"], d["answer"]))
            ids.append(d.get("id", i))
    return pairs, ids


def held_out_path(train_path):
    """The _heldout.jsonl beside a _train.jsonl. One naming rule, used by both arms."""
    if train_path.endswith("_train.jsonl"):
        return train_path[: -len("_train.jsonl")] + "_heldout.jsonl"
    return None


def pack_rows(pairs, tok, eos_id, seq):
    """Greedy pack into (seq+1)-token rows, prompt masked. Mirrors prepare_sft.pack_and_save.

    Returns (input_ids, labels, stats). One example never spans two rows; an example longer
    than a row is DROPPED, not truncated -- a truncated example's completion has no prompt
    for the mask to key on.
    """
    import torch

    row_len = seq + 1
    rows_i, rows_l = [], []
    cur_i, cur_l = [], []
    n_drop = n_pad = 0

    def flush():
        nonlocal cur_i, cur_l, n_pad
        if not cur_i:
            return
        pad = row_len - len(cur_i)
        n_pad += pad
        rows_i.append(cur_i + [eos_id] * pad)
        rows_l.append(cur_l + [-100] * pad)
        cur_i, cur_l = [], []

    # Batch-encode: one call per side beats 2N calls on a 500k-row pack.
    p_ids = tok([p for p, _ in pairs], add_special_tokens=False)["input_ids"]
    c_ids = tok([c for _, c in pairs], add_special_tokens=False)["input_ids"]
    for pi, ci in zip(p_ids, c_ids, strict=True):  # one per pair on both sides
        need = len(pi) + len(ci)
        if need > row_len:
            n_drop += 1
            continue
        if len(cur_i) + need > row_len:
            flush()
        cur_i += pi + ci
        cur_l += [-100] * len(pi) + list(ci)
    flush()
    stats = {"rows": len(rows_i), "dropped_overlong": n_drop, "pad_tokens": n_pad,
             "supervised": sum(1 for r in rows_l for v in r if v != -100),
             "total": len(rows_i) * row_len}
    return (torch.tensor(rows_i, dtype=torch.long),
            torch.tensor(rows_l, dtype=torch.long), stats)


def eval_loss(model, ids, lab, batch, device):
    """Mean loss over the validation rows. The number an lr scan is selected on."""
    import torch

    model.eval()
    tot = n = 0.0
    with torch.no_grad():
        for lo in range(0, ids.shape[0], batch):
            bi = ids[lo:lo + batch].to(device)
            bl = lab[lo:lo + batch].to(device)
            out = model(input_ids=bi[:, :-1], labels=bl[:, 1:])
            tot += out.loss.item() * bi.shape[0]
            n += bi.shape[0]
    model.train()
    return tot / max(n, 1)


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="data/controls/pythia-160m-step2000")
    ap.add_argument("--pack", default="data/sft/control_sft_text_train.jsonl")
    ap.add_argument("--out", default="ckpt_sft_pythia160m_control.pt")
    ap.add_argument("--epochs", type=int, default=1, help="sft_math.py's default")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=6, help="effective batch = batch * accum")
    ap.add_argument("--lr", type=float, default=1e-4,
                    help="AdamW lr. NOT derived from our Cfg -- see the LR note in the module "
                         "docstring. 1e-4 is where a CPU smoke test on this model actually "
                         "descends (11.52 -> 2.49 over 4 steps); 0.01 DIVERGES (-> 13.27)")
    ap.add_argument("--warmup", type=int, default=20, help="train.Cfg.warmup, absolute steps")
    ap.add_argument("--warmdown", type=float, default=0.65, help="train.Cfg.warmdown")
    ap.add_argument("--final_lr_frac", type=float, default=0.05, help="train.Cfg.final_lr_frac")
    ap.add_argument("--seq", type=int, default=SEQ)
    ap.add_argument("--max_steps", type=int, default=None, help="smoke tests only")
    ap.add_argument("--device", default="cuda",
                    help="'cpu' runs the real forward/backward without a card -- the only way "
                         "to prove this loop executes before it holds one. To pick a specific "
                         "card use CUDA_VISIBLE_DEVICES=<n> and leave this as 'cuda': both arms "
                         "run concurrently on different cards, and 'cuda:0' inside the process "
                         "would fight run_sft.sh's arm for card 0")
    ap.add_argument("--heldout", default=None,
                    help="the held-out text file. Defaults to the _heldout.jsonl beside the "
                         "--pack file; BOTH arms read the same two files so the held-out set "
                         "is one object, not two implementations of one rule")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_dir = a.model if os.path.isabs(a.model) else os.path.join(ROOT, a.model)
    pack_path = a.pack if os.path.isabs(a.pack) else os.path.join(ROOT, a.pack)
    for p in (model_dir, pack_path):
        if not os.path.exists(p):
            print(f"CANNOT RUN: {p} does not exist")
            return 2

    pack_sha = sha256_of(pack_path)
    print(f"pack {pack_path}\n  sha256 {pack_sha}", flush=True)

    tok = AutoTokenizer.from_pretrained(model_dir)
    # Our ChatML markers are single special tokens on our side; give this side dedicated
    # tokens too, so neither arm is handicapped on the stop symbol.
    vocab_before = len(tok)
    added = tok.add_special_tokens({"additional_special_tokens": ["<|im_start|>", IM_END]})
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # The ids the tokenizer assigned, recorded per fb's ruling: an id assigned at run time is
    # not reconstructable from the config afterwards, and a scorer that re-tokenizes with a
    # freshly built tokenizer would get different ones without any error.
    chatml_ids = {t: tok.convert_tokens_to_ids(t) for t in ("<|im_start|>", IM_END)}
    print(f"tokenizer {vocab_before} -> {len(tok)} entries (+{added} ChatML specials) "
          f"{chatml_ids}", flush=True)

    pairs, train_ids = read_pack(pack_path)
    hp = a.heldout or held_out_path(pack_path)
    val_pairs, val_ids = ([], [])
    if hp and os.path.exists(hp):
        val_pairs, val_ids = read_pack(hp)
    elif hp:
        print(f"CANNOT RUN: {hp} does not exist. The held-out file is written beside the train "
              f"file by build_control_sft_text.py; without it there is nothing to select an lr "
              f"on but training loss, which picks whichever lr memorised hardest.")
        return 2
    ids, lab, st = pack_rows(pairs, tok, tok.eos_token_id, a.seq)
    v_ids, v_lab, v_st = (pack_rows(val_pairs, tok, tok.eos_token_id, a.seq)
                          if val_pairs else (None, None, {}))
    if val_pairs:
        overlap = set(train_ids) & set(val_ids)
        if overlap:
            print(f"REFUSING: {len(overlap)} example id(s) appear in BOTH the train and "
                  f"held-out files, so the held-out loss is measured on trained data.")
            return 1
        print(f"validation {len(val_pairs):,} examples -> {v_st['rows']:,} rows, "
              f"{len(overlap)} overlap with training", flush=True)
    tokens = st["rows"] * (a.seq + 1)
    print(f"examples {len(pairs):,} -> rows {st['rows']:,} x {a.seq+1} = {tokens:,} tokens\n"
          f"  dropped over-length {st['dropped_overlong']:,}   pad {st['pad_tokens']:,}\n"
          f"  supervised {st['supervised']:,} of {st['total']:,} "
          f"({100.0*st['supervised']/max(st['total'],1):.1f}%)", flush=True)
    if st["supervised"] == 0 or st["supervised"] == st["total"]:
        print("REFUSING: the mask supervises nothing or everything")
        return 1

    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16)
    # NO resize. Pythia's config.vocab_size is 50,304 (padded for kernel alignment) while
    # its tokenizer holds 50,277 entries, so the two ChatML ids land at 50,277/50,278 --
    # INSIDE the existing embedding. resize_token_embeddings(len(tok)) would resize DOWN to
    # 50,279, dropping 25 rows and the 128-alignment: verified on the pod, 50,304 -> 50,279.
    #
    # THOSE ROWS ARE NOT TRAINED, and the earlier "would throw away 25 TRAINED rows" wording
    # is corrected here -- the same comment also called them "randomly-initialised in
    # effect", and both cannot hold. No tokenizer id maps to rows 50,254-50,303, so they
    # never received an embed_in gradient. Measured on the real checkpoint: those rows read
    # norm 0.6124 against 0.7733 for reachable rows (0.79x), and embed_out 0.8051 against
    # 1.0732. They are near-init, which is exactly what a fresh special token wants -- so
    # skipping the resize is still right, for the cost it actually has (25 rows and the
    # alignment) rather than one it does not.

    emb_rows = model.get_input_embeddings().weight.shape[0]
    max_id = max(chatml_ids.values())
    if max_id >= emb_rows:
        print(f"CANNOT RUN: ChatML id {max_id} is outside the embedding ({emb_rows} rows). "
              f"This model needs a resize UP -- add it deliberately rather than letting "
              f"resize_token_embeddings(len(tok)) also truncate.")
        return 2
    print(f"embedding {emb_rows} rows, ChatML ids at {sorted(chatml_ids.values())} -- inside, "
          f"no resize (a resize to len(tok)={len(tok)} would drop "
          f"{emb_rows - len(tok)} rows and the 128-alignment)", flush=True)
    model.gradient_checkpointing_enable()
    model.to(a.device).train()
    # fused AdamW is a CUDA-only kernel; the cpu path exists to smoke-test this loop.
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95),
                            weight_decay=0.01, fused=(a.device == "cuda"))
    for g in opt.param_groups:
        g["initial_lr"] = g["lr"]

    n = ids.shape[0]
    per_epoch = n // (a.batch * a.accum)
    total = per_epoch * a.epochs
    if a.max_steps:
        total = min(total, a.max_steps)
    print(f"steps {total} ({per_epoch}/epoch x {a.epochs}), effective batch "
          f"{a.batch*a.accum} rows", flush=True)
    if total < 1:
        print("REFUSING: fewer than one optimizer step -- batch*accum exceeds the pack")
        return 1

    # Same order both arms see: prepare_sft shuffles with seed 0 at pack time, and this pack
    # is already in that order, so read it straight through.
    # Measured BEFORE any step: an lr scan compares deltas, and a scan whose arms started
    # from different baselines would compare two different quantities.
    val_before = eval_loss(model, v_ids, v_lab, a.batch, a.device) if val_pairs else None
    if val_before is not None:
        print(f"held-out loss before training {val_before:.4f}", flush=True)

    t0 = time.time()
    step = 0
    losses = []
    for _ep in range(a.epochs):
        for s in range(per_epoch):
            if step >= total:
                break
            m = lr_mult_shape(step, total, a.warmup, a.warmdown, a.final_lr_frac)
            for g in opt.param_groups:
                g["lr"] = g["initial_lr"] * m
            opt.zero_grad(set_to_none=True)
            acc_loss = 0.0
            for k in range(a.accum):
                lo = (s * a.accum + k) * a.batch
                bi = ids[lo:lo + a.batch].to(a.device)
                bl = lab[lo:lo + a.batch].to(a.device)
                # train.py's convention: predict token t+1 from tokens <= t.
                out = model(input_ids=bi[:, :-1], labels=bl[:, 1:])
                (out.loss / a.accum).backward()
                acc_loss += out.loss.item() / a.accum
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(acc_loss)
            step += 1
            if step % 20 == 0 or step == 1:
                el = time.time() - t0
                print(f"  step {step}/{total}  loss {acc_loss:.4f}  lr {opt.param_groups[0]['lr']:.2e}"
                      f"  {el:.0f}s  eta {el/step*(total-step)/60:.0f}min", flush=True)

    val_after = eval_loss(model, v_ids, v_lab, a.batch, a.device) if val_pairs else None
    if val_after is not None:
        print(f"\nheld-out loss {val_before:.4f} -> {val_after:.4f} "
              f"(this is the number an lr scan is selected on)", flush=True)

    out_path = a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out)
    model.save_pretrained(out_path + ".hf")
    tok.save_pretrained(out_path + ".hf")
    meta = {
        "arm": "control", "base": a.model, "base_revision_note":
            "step2000 inferred from the download URL path; the config carries no step field",
        "pack": os.path.relpath(pack_path, ROOT), "pack_sha256": pack_sha,
        "examples": len(pairs), "rows": st["rows"], "seq": a.seq, "tokens": tokens,
        "dropped_overlong": st["dropped_overlong"],
        "vocab_before": vocab_before, "vocab_after": len(tok),
        "chatml_token_ids": chatml_ids,
        "embedding_rows": emb_rows,
        "resized": False,
        "resize_note": "config.vocab_size 50304 > tokenizer 50277, so the ChatML ids land "
                       "inside the existing embedding on rows no token addresses (norm "
                       "0.6124 vs 0.7733 for reachable rows: near-init, not trained); "
                       "resize_token_embeddings(len(tok)) would have shrunk it to 50279, "
                       "dropping 25 rows and the 128-alignment",
        "supervised_frac": round(st["supervised"] / max(st["total"], 1), 4),
        "steps": step, "epochs": a.epochs, "effective_batch": a.batch * a.accum,
        "optimizer": "AdamW (our arm uses Muon on 2D matrices -- NOT the same optimizer)",
        "lr": a.lr, "lr_note": "set by CPU smoke test on this model, NOT matched to our arm -- 1e-2 diverges here", "warmup": a.warmup, "warmdown": a.warmdown,
        "final_lr_frac": a.final_lr_frac,
        "final_train_loss": losses[-1] if losses else None,
        "first_train_loss": losses[0] if losses else None,
        # The selection number. Training loss is deliberately NOT it: selecting an lr on
        # training loss picks whichever lr memorised hardest, which is the artefact the scan
        # exists to rule out.
        "held_out_loss": val_after,
        "held_out_loss_before": val_before,
        "held_out_examples": len(val_pairs),
        "held_out_ids_sha256": hashlib.sha256(
            ",".join(str(i) for i in val_ids).encode()).hexdigest()[:16],
        # Per supervised BYTE, so the two arms' held-out losses are comparable despite
        # different tokenizers: loss-per-token is not a shared unit when the tokenizers
        # segment the same text into different counts.
        # IMPORTED, not recomputed: two implementations of one denominator is exactly the
        # drift the shared-split ruling exists to prevent, and the arms' per-byte losses are
        # only comparable if the divisor is literally the same function.
        "held_out_supervised_bytes": (
            eval_heldout_ours.supervised_bytes(hp)[0] if hp and os.path.exists(hp) else 0),
        "wall_s": round(time.time() - t0),
    }
    with open(out_path + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"\nsaved {out_path}.hf  meta {out_path}.meta.json")
    print(json.dumps(meta, indent=2))
    return 0


def selftest():
    """The three things here that could be wrong without any error appearing.

    No card, no model download: a fake tokenizer stands in, because what is being checked is
    the mask, the packing and the schedule -- none of which involve weights.
    """
    fails = []

    # 1. the LR schedule must equal train.lr_mult exactly, since the whole point of
    #    re-expressing it is that the two arms share a shape.
    try:
        sys.path.insert(0, ROOT)
        from train import Cfg, lr_mult
        for total in (100, 733):
            for step in (0, 1, 19, 20, 21, total // 2, total - 2, total - 1, total, total + 5):
                mine = lr_mult_shape(step, total, Cfg.warmup, Cfg.warmdown, Cfg.final_lr_frac)
                theirs = lr_mult(step, total, Cfg)
                if abs(mine - theirs) > 1e-12:
                    fails.append(f"schedule differs at step {step}/{total}: {mine} vs {theirs}")
                    break
    except Exception as e:  # noqa: BLE001
        fails.append(f"could not compare against train.lr_mult: {type(e).__name__}: {e} "
                     f"-- the shape claim is then UNVERIFIED, not verified")

    # 2. the mask: prompt tokens -100, completion tokens supervised and equal to the input.
    try:
        import torch  # noqa: F401

        class FakeTok:
            """One id per whitespace token; ids are stable per word so labels are checkable."""
            def __init__(self):
                self.vocab = {}

            def __call__(self, texts, add_special_tokens=False):
                out = []
                for t in texts:
                    ids = []
                    for w in t.split():
                        ids.append(self.vocab.setdefault(w, len(self.vocab) + 10))
                    out.append(ids)
                return {"input_ids": out}

        ft = FakeTok()
        pairs = [("p1 p2 p3", "c1 c2"), ("q1", "d1 d2 d3")]
        ids, lab, st = pack_rows(pairs, ft, 0, 16)
        sup = (lab != -100)
        if not bool((lab[sup] == ids[sup]).all()):
            fails.append("a supervised label differs from its input token")
        # exactly the completion tokens are supervised: 2 + 3
        if int(sup.sum()) != 5:
            fails.append(f"expected 5 supervised tokens, got {int(sup.sum())}")
        # and the first prompt token of row 0 is masked
        if lab[0, 0] != -100:
            fails.append("the first prompt token is supervised")
        # an over-length example is dropped, never truncated
        _, _, st2 = pack_rows([("x " * 40, "y")], ft, 0, 8)
        if st2["dropped_overlong"] != 1 or st2["rows"] != 0:
            fails.append(f"over-length example not dropped: {st2}")
    except Exception as e:  # noqa: BLE001
        fails.append(f"mask check could not run: {type(e).__name__}: {e}")

    # 3. our template must actually put the stop token in the SUPERVISED half.
    p, c = format_example("q", "a")
    if not c.endswith(IM_END):
        fails.append(f"the completion does not end with the stop token: {c!r}")
    if IM_END in p.split("<|im_start|>assistant")[-1]:
        fails.append("the prompt's trailing assistant marker is followed by a stop token")

    # 4. the split is now made ONCE by the builder, so what this arm must get right is
    #    reading it: the id field, and the _train -> _heldout name mapping. The old version of
    #    this case tested an in-arm holdout_every split, which no longer exists -- a test of a
    #    deleted code path passes forever and proves nothing.
    import tempfile as _tf
    with _tf.TemporaryDirectory() as d:
        tp = os.path.join(d, "pack_train.jsonl")
        hp = os.path.join(d, "pack_heldout.jsonl")
        with open(tp, "w", encoding="utf-8") as f:
            for i in (1, 2, 3, 4):
                f.write(json.dumps({"id": i, "question": f"q{i}", "answer": f"a{i}"}) + "\n")
        with open(hp, "w", encoding="utf-8") as f:
            f.write(json.dumps({"id": 0, "question": "q0", "answer": "a0"}) + "\n")
        if held_out_path(tp) != hp:
            fails.append(f"held_out_path({tp}) gave {held_out_path(tp)}, expected {hp}")
        tr, tr_ids = read_pack(tp)
        va, va_ids = read_pack(hp)
        if tr_ids != [1, 2, 3, 4] or va_ids != [0]:
            fails.append(f"ids read wrong: train {tr_ids}, held-out {va_ids}")
        if set(tr_ids) & set(va_ids):
            fails.append("the two files share an example id")
        if len(tr) != 4 or len(va) != 1:
            fails.append(f"pair counts wrong: {len(tr)}, {len(va)}")
        # a file with no id field must fall back to line order, not crash
        np_ = os.path.join(d, "noid_train.jsonl")
        with open(np_, "w", encoding="utf-8") as f:
            f.write(json.dumps({"question": "q", "answer": "a"}) + "\n")
        if read_pack(np_)[1] != [0]:
            fails.append("a row without an id field did not fall back to line order")
        if held_out_path("/x/plain.jsonl") is not None:
            fails.append("held_out_path invented a path for a non-_train filename")

    # 5. THE RESIZE MUST STAY GONE. resize_token_embeddings(len(tok)) SHRINKS this model
    #    (50,304 -> 50,279, verified on the pod) and says nothing while doing it: no error,
    #    nothing downstream goes red, just 25 rows and the 128-alignment gone. So absence of
    #    the call is the only thing left to check. The needle is the CALL -- ".resize_token_"
    #    + "embeddings(" -- which the prose in this file deliberately never writes that way,
    #    so it cannot match its own documentation.
    #
    #    A SECOND grep was written beside it and DELETED before commit: it asserted the
    #    bounds-check message above was still present, but that message appears on the check's
    #    own line, so `needle not in src` was false by construction. It passed its broken
    #    world because it could not fail, not because the code was intact -- de's shape from
    #    this morning, a criterion whose needle lands in its own data matching itself.
    src = open(os.path.abspath(__file__), encoding="utf-8").read()
    if ".resize_token_" + "embeddings(" in src:
        fails.append("resize_token_embeddings is called again: on this model it SHRINKS "
                     "50304 -> 50279 silently. Grow to a multiple of 128, or do nothing.")

    for f in fails:
        print(f"  SELFTEST FAIL {f}")
    if fails:
        print(f"\n{len(fails)} selftest failure(s)")
        return 1
    print("sft_hf_control selftest OK (schedule == train.lr_mult, mask exact, stop supervised, "
          "shared held-out file read with matching ids)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
