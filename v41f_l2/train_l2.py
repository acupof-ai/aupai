"""L2 quality head training entry point.

Wires the encoder + QualityHead (l2_quality_head.py) to de's ledger loader
(datagen/l2_dataset.py, PR #397) and runs the MSE + within-domain-rank objective.

Flow:
    load_pairs(ledger, pool) -> split_pairs (doc-level, leak-free)
    build_vocab over train+val domains -> make_torch_loader (shared domain ids)
    frozen encoder -> pool -> 4-dim head -> quality_loss (per-dim MSE + domain rank)

The encoder choice (--encoder) defaults to the probe winner bge-m3 (multilingual, 8192
ctx, near-zero truncation on real docs); Qwen3-Emb-0.6B is the alternative (last-token).
This script is ready to run once de's loader is on main and 66's labelled ledger lands;
it is intentionally thin — the tested math lives in l2_quality_head.py.
"""

from __future__ import annotations

import argparse

import torch

from v41f_l2.l2_quality_head import L2Config, L2QualityModel, quality_loss, save_head


def load_encoder(encoder: str, device: str):
    """Returns (hf_model, pooling_mode). Pooling matches each model's shipped config."""
    from transformers import AutoModel

    last_token = {"Qwen/Qwen3-Embedding-0.6B", "qwen3-emb-0.6b"}
    model = AutoModel.from_pretrained(
        encoder, torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    return model, ("lasttoken" if encoder in last_token else "cls")


def build_loaders(args, tokenizer):
    # soft dependency: de's loader is the single data path; until #397 merges this fails loud
    try:
        from datagen.l2_dataset import build_vocab, load_pairs, make_torch_loader, split_pairs
    except ImportError as e:
        raise SystemExit(
            "datagen/l2_dataset.py not found (PR #397). Merge it to main before training."
        ) from e

    pairs = load_pairs(args.ledger, args.pool, scorer_name="l3-rubric", rubric_kind=args.rubric_kind)
    train, val = split_pairs(pairs, val_frac=args.val_frac)
    # doc-hash split could put every doc of a rare kind in one side; a kind seen overall but
    # missing from a split means the model never trains (or never evals) it — fail loud.
    kinds_all = {ex.rubric_kind for ex in pairs}
    for name, split in (("train", train), ("val", val)):
        missing = kinds_all - {ex.rubric_kind for ex in split}
        if missing:
            raise SystemExit(
                f"{name} split is empty for rubric_kind(s) {sorted(missing)}; reseed or lower val_frac"
            )
    # shared domain vocab across both splits: same domain must get the same id, and an
    # unseen domain at batch time raises inside the loader rather than becoming a silent -1.
    domain_vocab = build_vocab(ex.domain for ex in train + val)
    common = dict(
        encode=tokenizer.encode,
        batch_size=args.batch_size,
        max_len=args.max_len,
        pad_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0,
        domain_vocab=domain_vocab,
    )
    dl_tr = make_torch_loader(train, shuffle=True, **common)
    dl_va = make_torch_loader(val, shuffle=False, **common)
    return dl_tr, dl_va


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--encoder", default="BAAI/bge-m3")
    ap.add_argument("--rubric-kind", default=None, choices=[None, "code", "natural_language"])
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lambda-rank", type=float, default=0.5)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--save-ckpt", help="write the trained QualityHead state_dict here")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.encoder)
    enc, pooling = load_encoder(args.encoder, args.device)
    cfg = L2Config(lambda_rank=args.lambda_rank)
    model = L2QualityModel(enc.to(args.device), cfg, pooling=pooling).to(args.device)
    dl_tr, dl_va = build_loaders(args, tok)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    for ep in range(args.epochs):
        model.train()
        last = None
        for batch in dl_tr:
            ids = batch["input_ids"].to(args.device)
            attn = batch["attention_mask"].to(args.device)
            labels = batch["labels"].to(args.device)
            mask = batch["label_mask"].to(args.device)
            dom = batch["domain_id"].to(args.device)
            pred = model(ids, attn)
            loss, parts = quality_loss(pred, labels, mask, dom, cfg)
            opt.zero_grad()
            loss.backward()
            opt.step()
            last = parts
        print(f"epoch {ep} train {last}")  # real per-epoch val aggregation added with data

    # head only (frozen public encoder is never checkpointed); scanner loads this exact file
    if args.save_ckpt:
        save_head(args.save_ckpt, model.head, cfg)
        print(f"saved head checkpoint -> {args.save_ckpt}")


if __name__ == "__main__":
    main()
