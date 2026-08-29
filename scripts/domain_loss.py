#!/usr/bin/env python3
"""Held-out loss for one checkpoint, reported per domain.

train.py reports a single combined figure, and on a mix that is 49.6% textbook that
figure is largely a textbook figure. Per domain it becomes two things at once: the
backbone of a score matrix that works on a base checkpoint, and the only way to price
a domain -- "is a CCI3 token worth a web_hq token" is this number and nothing else.

Runs on a checkpoint after the fact, so the six 0830v1 budget points can be filled in
without retraining. Scores with the vocabulary the checkpoint was trained on.

    python scripts/domain_loss.py --ckpt ckpt_0830v1_3.24b.pt [--mix data/mix_scale_3.24b.json]
    python scripts/domain_loss.py --ckpt A.pt --ckpt B.pt --json runs/domain_loss.json
"""

import argparse
import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.loader import EOS_ID, load_checkpoint, load_tokenizer  # noqa: E402

# The same held-out rows every time, drawn the same way for every checkpoint and every
# domain. A per-domain number is only comparable across checkpoints if the rows are
# identical, so the split is a fixed tail of each domain's shards, never a sample.
HOLDOUT_ROWS = 256
SEQ_CAP = 64  # sequences per domain: 64 x 4096 = 262K tokens, enough for +-0.01 nat


def domain_files(mix_path, root):
    mix = json.load(open(mix_path, encoding="utf-8"))
    out = {}
    for name in mix["domains"]:
        d = os.path.join(root, "data", "corpus", name)
        files = sorted(f for f in os.listdir(d) if f.endswith(".jsonl")) if os.path.isdir(d) else []
        if not files:
            print(f"  {name}: no shards under {d} -- SKIPPED, not scored as zero", flush=True)
            continue
        # The LAST shard, and the tail of it: train.py consumes shards in order from the
        # front, so the tail of the last shard is the part a budget-capped run never read.
        out[name] = os.path.join(d, files[-1])
    return out


def tail_texts(path, n):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(line)
            if len(rows) > n * 4:  # keep a rolling tail without holding the whole shard
                rows = rows[-n * 2 :]
    texts = []
    for line in rows[-n:]:
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        t = d.get("content") or d.get("text") or ""
        if t:
            texts.append(t)
    return texts


@torch.no_grad()
def domain_loss(model, tok, texts, seq, device):
    """Mean next-token CE over packed held-out text. Packing matches training, so the
    number is on the same scale as the val figure train.py prints."""
    ids = []
    for t in texts:
        ids.extend(tok.encode(t).ids + [EOS_ID])
    n = (len(ids) - 1) // seq
    if n == 0:
        return None, 0
    n = min(n, SEQ_CAP)
    x = torch.tensor(ids[: n * seq], dtype=torch.long).view(n, seq).to(device)
    y = torch.tensor(ids[1 : n * seq + 1], dtype=torch.long).view(n, seq).to(device)
    tot = cnt = 0.0
    for i in range(0, n, 4):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
            logits = model(x[i : i + 4])
        if isinstance(logits, tuple):
            logits = logits[0]
        b = logits.shape[0]
        loss = torch.nn.functional.cross_entropy(
            logits.float().view(-1, logits.shape[-1]), y[i : i + 4].reshape(-1), reduction="sum"
        )
        tot += loss.item()
        cnt += b * seq
    return tot / cnt, int(cnt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True)
    ap.add_argument("--mix", default=os.path.join(ROOT, "data/mix_scale_3.24b.json"))
    ap.add_argument("--tokenizer", default=os.path.join(ROOT, "data/tokenizer.json"))
    ap.add_argument("--json", help="append one record per checkpoint here")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    files = domain_files(a.mix, ROOT)
    assert files, f"{a.mix} named domains but none have shards -- nothing to score"
    cache = {name: tail_texts(p, HOLDOUT_ROWS) for name, p in files.items()}

    out = []
    for ck_path in a.ckpt:
        # bf16: the MLA path goes through FlashAttention, which refuses fp32 outright.
        model, cfg = load_checkpoint(ck_path, device=device, dtype=torch.bfloat16)
        # load_tokenizer cross-checks size and vocab_id against this cfg and raises on a
        # mismatch. Scoring with the wrong vocabulary is silent noise, so it is checked
        # per checkpoint rather than once for the batch.
        tok = load_tokenizer(a.tokenizer, cfg)
        model.eval()
        seq = getattr(cfg, "seq", 4096)  # cfg is a SimpleNamespace, not a dict
        row = {"ckpt": os.path.basename(ck_path), "domains": {}}
        print(f"\n{os.path.basename(ck_path)}  (vocab {getattr(cfg, 'vocab', '?')}, seq {seq})", flush=True)
        for name, texts in cache.items():
            loss, ntok = domain_loss(model, tok, texts, seq, device)
            if loss is None:
                print(f"  {name:10s} too few tokens to score -- SKIPPED", flush=True)
                continue
            row["domains"][name] = {"loss": round(loss, 4), "tokens": ntok}
            print(f"  {name:10s} {loss:.4f}   ({ntok:,} tok)", flush=True)
        vals = [d["loss"] for d in row["domains"].values()]
        row["unweighted_mean"] = round(sum(vals) / len(vals), 4)
        # NOT the mix-weighted figure train.py prints: an unweighted mean asks "how good
        # across domains", the weighted one asks "how good on this mix". Reporting the
        # weighted one alone is what let a 49.6%-textbook mix read as a model result.
        print(f"  {'MEAN':10s} {row['unweighted_mean']:.4f}   (unweighted across domains)", flush=True)
        out.append(row)
        del model
        torch.cuda.empty_cache()

    if a.json:
        with open(a.json, "a", encoding="utf-8") as f:
            for r in out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\nappended {len(out)} record(s) to {a.json}")


if __name__ == "__main__":
    main()
