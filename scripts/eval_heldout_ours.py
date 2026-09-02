#!/usr/bin/env python3
# restartable: loads a checkpoint and a pack, prints one number, writes nothing. An interrupt
# costs a rerun.
"""Held-out loss of OUR checkpoint on the shared control held-out pack. One card, minutes.

    python3 scripts/eval_heldout_ours.py \
        --ckpt ckpt_sft_control_ours.pt \
        --pack data/sft/control_sft_ours_heldout.pt

WHY THIS EXISTS. sft_math.py does not evaluate a held-out set -- it trains and stops. The
control arm reports held_out_loss (scripts/sft_hf_control.py), so without this our arm has
no comparable number and the two headline figures would be a held-out loss on one side and
nothing on the other. fb froze our recipe, so this is a separate reader rather than an edit
to sft_math.py: it loads a finished checkpoint and scores it.

THE UNIT IS LOSS PER SUPERVISED BYTE, not per token. The two arms tokenize the same text
with different tokenizers, so a per-token loss is not a shared unit -- the same held-out
text is a different number of tokens on each side, and dividing by each side's own count
compares two different quantities (docs/lessons/gate_failure_shapes.md §64). Bytes are the
one denominator both arms agree on, and only the SUPERVISED bytes count because prompts are
masked. This file prints per-token as well, labelled as not comparable across arms, because
it is the number that matches the training log.
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)


def supervised_bytes(text_path):
    """Bytes of the completions in the held-out TEXT -- the cross-arm denominator.

    Read from the text, not reconstructed from token ids: decoding ids back to text would
    measure the tokenizer's round-trip rather than the source, and the point of this
    denominator is that it does not depend on the tokenizer.
    """
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from loader import format_example

    total = 0
    n = 0
    with open(text_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            _, completion = format_example(d["question"], d["answer"])
            total += len(completion.encode("utf-8"))
            n += 1
    return total, n


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    # Not required=True: that would make --selftest unreachable, and a selftest you cannot
    # invoke is not gated by check_selftests_are_gated, it is merely listed there.
    ap.add_argument("--ckpt", help="an SFT'd checkpoint of OURS")
    ap.add_argument("--pack", default=os.path.join(ROOT, "data", "sft",
                                                   "control_sft_ours_heldout.pt"))
    ap.add_argument("--text", default=None,
                    help="the held-out TEXT file, for the supervised-byte denominator. "
                         "Defaults to data/sft/control_sft_text_heldout.jsonl")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--json_out", default=None, help="also write the numbers here")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.ckpt:
        ap.error("need --ckpt, or --selftest")

    text = a.text or os.path.join(ROOT, "data", "sft", "control_sft_text_heldout.jsonl")
    for p in (a.ckpt, a.pack, text):
        if not os.path.exists(p):
            print(f"CANNOT CHECK: {p} does not exist")
            return 2

    import torch

    from train import Cfg, Transformer  # noqa: F401  -- Cfg is mutated by the checkpoint

    d = torch.load(a.pack, map_location="cpu", weights_only=True)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)

    # The same refusal sft_math.py makes: a pack from another vocabulary scores every id
    # valid and wrong, and the loss would look merely bad rather than meaningless.
    ck_vocab = ck.get("vocab_id")
    if ck_vocab and d.get("vocab_id") and d["vocab_id"] != ck_vocab:
        print(f"REFUSING: pack vocab_id {d['vocab_id']} != checkpoint {ck_vocab}")
        return 1

    cfg = ck.get("cfg", {})
    for k, v in cfg.items():
        if hasattr(Cfg, k):
            setattr(Cfg, k, v)
    model = Transformer(Cfg).to(a.device)
    sd = ck.get("model") or ck.get("state_dict") or ck
    missing, unexpected = model.load_state_dict(
        {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in sd.items()},
        strict=False)
    if missing:
        print(f"REFUSING: {len(missing)} parameter(s) missing from the checkpoint, e.g. "
              f"{missing[:3]} -- a partially loaded model's loss is not this model's loss")
        return 1
    if unexpected:
        print(f"note: {len(unexpected)} unexpected key(s) ignored, e.g. {unexpected[:3]}")
    model.eval()

    X = d["input_ids"][:, :-1].long()
    Y = d["labels"][:, 1:].long()
    tot_loss = 0.0
    tot_tok = 0
    with torch.no_grad():
        for lo in range(0, X.shape[0], a.batch):
            x = X[lo:lo + a.batch].to(a.device)
            y = Y[lo:lo + a.batch].to(a.device)
            logits = model(x)
            n = int((y != -100).sum())
            if n == 0:
                continue
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(), y.reshape(-1),
                ignore_index=-100, reduction="sum")
            tot_loss += loss.item()
            tot_tok += n

    if tot_tok == 0:
        print("CANNOT CHECK: no supervised token in the held-out pack")
        return 2

    sbytes, n_ex = supervised_bytes(text)
    per_token = tot_loss / tot_tok
    per_byte = tot_loss / sbytes
    out = {
        "ckpt": os.path.basename(a.ckpt), "pack": os.path.basename(a.pack),
        "held_out_examples": n_ex, "rows": int(X.shape[0]),
        "supervised_tokens": tot_tok, "supervised_bytes": sbytes,
        "total_nll": tot_loss,
        "held_out_loss_per_supervised_byte": per_byte,
        "held_out_loss_per_token": per_token,
        "note": "per_supervised_byte is the cross-arm comparable number; per_token is NOT "
                "comparable to the control arm, whose tokenizer segments the same text "
                "differently",
    }
    print(f"held-out examples          {n_ex:,} ({X.shape[0]:,} packed rows)")
    print(f"supervised tokens          {tot_tok:,}")
    print(f"supervised bytes           {sbytes:,}")
    print(f"loss / supervised BYTE     {per_byte:.6f}   <- compare this across arms")
    print(f"loss / token               {per_token:.6f}   (not cross-arm comparable)")
    if a.json_out:
        with open(a.json_out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"wrote {a.json_out}")
    return 0


def selftest():
    """The denominator, and the refusals. No card, no checkpoint."""
    import json as _json
    import tempfile

    fails = []
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from loader import format_example

    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "h.jsonl")
        rows = [{"id": 0, "question": "q", "answer": "a"},
                {"id": 5, "question": "问题", "answer": "答案"}]
        with open(p, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(_json.dumps(r, ensure_ascii=False) + "\n")
        got, n = supervised_bytes(p)
        want = sum(len(format_example(r["question"], r["answer"])[1].encode("utf-8"))
                   for r in rows)
        if got != want or n != 2:
            fails.append(f"supervised_bytes {got}/{n}, expected {want}/2")
        # A multibyte answer must count MORE bytes than characters, or the denominator is
        # silently character-based and the two arms' numbers would still not be comparable.
        one, _ = supervised_bytes(p)
        ascii_only = os.path.join(d, "a.jsonl")
        with open(ascii_only, "w", encoding="utf-8") as f:
            f.write(_json.dumps({"question": "q", "answer": "aa"}) + "\n")
            f.write(_json.dumps({"question": "q", "answer": "aa"}) + "\n")
        two, _ = supervised_bytes(ascii_only)
        if not one > two:
            fails.append(f"multibyte text did not count more bytes: {one} vs {two}")

    for f in fails:
        print(f"  SELFTEST FAIL {f}")
    if fails:
        print(f"\n{len(fails)} selftest failure(s)")
        return 1
    print("eval_heldout_ours selftest OK (supervised-byte denominator counts bytes, not chars)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
