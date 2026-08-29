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

# train.py:1187 holds out the HEAD of each domain -- seqs[:n_val] is validation, seqs[n_val:]
# is the training pool -- so the head is unseen for EVERY budget by construction. Scoring the
# tail instead reads the training pool, and reads more of it the larger the budget: at epoch cap
# 1 the 3.24B run consumes essentially the whole pool while the 0.2B run barely touches it. That
# hands larger budgets an easier test set and inflates exactly the deltas this script exists to
# report. Score the head.
HOLDOUT_ROWS = 4000  # source lines read from the first shard, packed then truncated to SEQ_CAP
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
        out[name] = os.path.join(d, files[0])  # the head, matching train.py's val split
    return out


def head_texts(path, n):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(line)
            if len(rows) >= n:
                break
    texts = []
    for line in rows:
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        t = d.get("content") or d.get("text") or ""
        if t:
            texts.append(t)
    return texts


@torch.no_grad()
def domain_loss(model, tok, texts, seq, device, cap=SEQ_CAP):
    """Mean next-token CE over packed held-out text. Packing matches training, so the
    number is on the same scale as the val figure train.py prints."""
    ids = []
    for t in texts:
        ids.extend(tok.encode(t).ids + [EOS_ID])
    n = (len(ids) - 1) // seq
    if n == 0:
        return None, 0
    n = min(n, cap)
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


def selftest(model, tok, texts, seq, device):
    """Three known answers. A metric without one is not a metric -- four numbers in this
    repo were wrong for a day because nobody had a case where they must fail."""
    import random

    real, _ = domain_loss(model, tok, texts, seq, device)

    # 1. SHUFFLED text must score far worse. Same tokens, same length, no structure:
    #    a scorer that reads its input at all cannot be indifferent to this.
    ids = [i for t in texts for i in tok.encode(t).ids]
    random.Random(0).shuffle(ids)
    shuf = ["".join(tok.decode([i]) for i in ids[: len(ids)])]
    bad, _ = domain_loss(model, tok, shuf, seq, device)

    # 2. Scale invariance across the accumulator: the same text twice, scored over twice as
    #    many batches, must give the same PER-TOKEN mean. `texts * N` alone cannot test this --
    #    the cap truncates it back to the same N sequences and the assertion passes vacuously,
    #    which is how this selftest was first written. The cap has to move with the input.
    a, _ = domain_loss(model, tok, texts, seq, device, cap=16)
    b, _ = domain_loss(model, tok, texts * 2, seq, device, cap=32)

    ok = True
    print(f"  selftest real {real:.4f} | shuffled {bad:.4f} | 16seq {a:.4f} | 32seq(2x) {b:.4f}")
    if bad - real < 1.0:
        print(f"  FAIL shuffled text scores {bad - real:+.3f} vs real; must be much worse")
        ok = False
    if abs(b - a) > 0.05:
        print(
            f"  FAIL doubling the input moved a per-token mean by {b - a:+.3f}; the accumulator "
            "is size-dependent"
        )
        ok = False
    print("  selftest " + ("OK" if ok else "FAILED"))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True)
    ap.add_argument("--mix", default=os.path.join(ROOT, "data/mix_scale_3.24b.json"))
    ap.add_argument("--tokenizer", default=os.path.join(ROOT, "data/tokenizer.json"))
    ap.add_argument("--json", help="append one record per checkpoint here")
    ap.add_argument("--selftest", action="store_true", help="known answers; run before believing any number")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    files = domain_files(a.mix, ROOT)
    assert files, f"{a.mix} named domains but none have shards -- nothing to score"
    cache = {name: head_texts(p, HOLDOUT_ROWS) for name, p in files.items()}

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
        if a.selftest:
            probe = next(iter(cache.values()))
            if not selftest(model, tok, probe, seq, device):
                sys.exit("selftest failed -- the numbers below would not be measurements")
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
