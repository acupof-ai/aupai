#!/usr/bin/env python3
"""Val loss on the held-out signature-only control pack (prereg format_sft_humaneval_0909).

The control pairs (data/sft/code_if_pairs_control.jsonl) never enter training:
the format-SFT packer filters them out of the train pack, and this script
scores a 1,000-pair slice packed with the same split-encode boundary
(sft_format_val_0909.pt). Loss on them at init vs after SFT is the
negative-control readout: if the docstring arm moves but sig-only loss does
not, the model learned the docstring signal, not function-body continuation.

Loss path mirrors sft_math.py exactly -- same model call (idx, targets,
cu_seqlens), same LigerFusedLinearCrossEntropyLoss with the same softcap -- so
the init and post-SFT numbers are comparable to each other and to training loss.

Usage:
    python3 eval/sft_val_loss.py --ckpt <ckpt> --pack data/sft/sft_format_val_0909.pt
"""

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
sys.path.insert(0, ROOT)  # train
sys.path.insert(0, os.path.join(ROOT, "scripts"))  # loader

import torch  # noqa: E402
from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss  # noqa: E402

from train import SOFTCAP, doc_cu_seqlens  # noqa: E402

EOS_ID = 1  # <eos> id in data/tokenizer.json
BATCH = 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        sys.exit("REFUSING: CUDA_VISIBLE_DEVICES is unset, so cuda:0 is physical "
                 "GPU 0 -- tileRL's card. Set it to your granted card.")

    from loader import load_checkpoint, vocab_fingerprint
    from tokenizers import Tokenizer

    d = torch.load(args.pack, weights_only=True)
    tok = Tokenizer.from_file(os.path.join(ROOT, "data", "tokenizer.json"))
    pack_vocab = d.get("vocab_id")
    assert pack_vocab == vocab_fingerprint(tok), (
        f"{args.pack} was packed against vocabulary {pack_vocab} but the tokenizer "
        f"is {vocab_fingerprint(tok)} -- the loss would be over shifted ids")

    model, cfg = load_checkpoint(args.ckpt, device=args.device)
    model.eval()
    ids, lab = d["input_ids"].to(args.device), d["labels"].to(args.device)
    n_rows = ids.shape[0]
    flce = LigerFusedLinearCrossEntropyLoss(ignore_index=-100, softcap=SOFTCAP)
    weight = model.head.weight[: model.cfg.vocab]
    use_doc_mask = bool(getattr(cfg, "doc_mask", False))

    # flce reduces mean over supervised tokens per batch, so weight each batch
    # by its supervised count for the global mean.
    tot, ntok = 0.0, 0
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for i in range(0, n_rows, BATCH):
            xb = ids[i:i + BATCH, :-1]
            yb = lab[i:i + BATCH, 1:]
            cub = doc_cu_seqlens(xb, EOS_ID) if use_doc_mask else None
            hidden, _ = model(xb, yb, cub, None)
            B, T, H = hidden.shape
            loss = flce(weight, hidden.to(weight.dtype).reshape(-1, H), yb.reshape(-1))
            n = int((yb != -100).sum())
            tot += loss.item() * n
            ntok += n
    print(f"val loss = {tot / ntok:.4f} over {ntok} supervised tokens "
          f"({n_rows} rows, {os.path.basename(args.pack)})", flush=True)


if __name__ == "__main__":
    main()
