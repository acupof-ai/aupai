#!/usr/bin/env python3
"""Does FoNE actually teach arithmetic? A/B on synthetic sums, same budget both ways.

The claim under test is narrow and checkable: k5 writes perfectly formatted chains
whose arithmetic is invented (160 - 8 = 320), and the suspected cause is that the
BPE vocab splits numbers by corpus frequency rather than place value -- 1640 as
16|40, 3200 as 3|200 -- so a carry rule learned on one number cannot transfer to
the next. If that is the cause, giving the model the VALUE instead of the fragments
should move held-out accuracy a lot, at equal parameters and equal steps.

Deliberately small: a few minutes on one GPU, or CPU with the fla stand-in. It
answers whether the full pretrain is worth its hours, so it runs first.

  python scripts/fone_probe.py [--steps 800] [--device cuda]
"""

import argparse
import os
import random
import sys

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("FLA_FLASH_KDA", "0")

import fone  # noqa: E402
import train  # noqa: E402

if train.chunk_kda is None:  # CPU box without fla: shape-preserving stand-in
    train.chunk_kda = lambda q, k, v, **kw: (q * 0 + v, None)


def problems(n, rng, lo=10, hi=99):
    """a + b = c and a - b = c over two-digit operands. Held-out pairs are unseen."""
    out = []
    for _ in range(n):
        a, b = rng.randint(lo, hi), rng.randint(lo, hi)
        if rng.random() < 0.5:
            out.append((f"{a}+{b}=", a + b))
        else:
            a, b = max(a, b), min(a, b)
            out.append((f"{a}-{b}=", a - b))
    return out


def make_rows(pairs, tok, seq, use_fone, num_id):
    """Problems -> (ids, in-values, target-values, answer mask), one problem per row."""
    ids = torch.zeros(len(pairs), seq, dtype=torch.long)
    vin = torch.zeros(len(pairs), seq)
    vtg = torch.zeros(len(pairs), seq)
    ans = torch.zeros(len(pairs), seq, dtype=torch.bool)
    for i, (q, c) in enumerate(pairs):
        if use_fone:
            pieces, vals = fone.encode_text([q + str(c)], tok, num_id)
            row = pieces[0].tolist()
            vi = []
            k = 0
            for t in row:
                vi.append(vals[k] if t == num_id else 0.0)
                k += t == num_id
        else:
            row = tok.encode(q + str(c)).ids
            vi = [0.0] * len(row)
        row = row[:seq]
        n = len(row)
        ids[i, :n] = torch.tensor(row)
        vin[i, :n] = torch.tensor(vi[:n])
        # the answer is whatever follows '='; supervise only those positions
        eq = tok.encode(q).ids
        if use_fone:
            eq = fone.encode_text([q], tok, num_id)[0][0].tolist()
        ans[i, len(eq) - 1 : n - 1] = True
        if use_fone:
            vtg[i, len(eq) - 1 : n - 1] = float(c)
    return ids, vin, vtg, ans


def run(use_fone, steps, device, seed=0):
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(train.TOK_PATH)
    num_id = tok.token_to_id(fone.NUM_TOKEN)
    assert num_id is not None, "tokenizer has no [NUM]; run scripts/build_tokenizer.py"

    cfg = type("C", (), {k: v for k, v in vars(train.Cfg).items() if not k.startswith("_")})
    cfg.d, cfg.layers, cfg.heads, cfg.head_dim, cfg.ffn_hidden = 256, 4, 4, 64, 512
    cfg.seq, cfg.attn_window, cfg.grad_ckpt, cfg.attn_res = 32, 32, False, False
    cfg.fone, cfg.num_id = use_fone, num_id

    rng = random.Random(seed)
    tr = problems(20000, rng)
    seen = {q for q, _ in tr}
    te = [p for p in problems(4000, random.Random(seed + 1)) if p[0] not in seen][:1000]

    Xtr, Vtr, Wtr, Atr = make_rows(tr, tok, cfg.seq, use_fone, num_id)
    Xte, Vte, Wte, Ate = make_rows(te, tok, cfg.seq, use_fone, num_id)

    torch.manual_seed(seed)
    m = train.HybridLM(cfg).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
    B = 64
    for s in range(steps):
        i = torch.randint(0, len(Xtr) - B, (1,)).item()
        x, a = Xtr[i : i + B].to(device), Atr[i : i + B].to(device)
        v = Vtr[i : i + B].to(device) if use_fone else None
        h, _ = m(x[:, :-1], torch.zeros(1), None, v[:, :-1] if use_fone else None)
        y = x[:, 1:].clone()
        y[~a[:, :-1]] = -100
        logits = m.head(h)[..., : cfg.vocab].float()
        loss = F.cross_entropy(logits.reshape(-1, cfg.vocab), y.reshape(-1), ignore_index=-100)
        if use_fone:
            nm = (y == num_id) & a[:, :-1]
            if nm.any():
                w = Wtr[i : i + B].to(device)[:, 1:]
                loss = loss + F.cross_entropy(
                    m.num_logits(h[nm].float()).reshape(-1, 10), fone.digit_targets(w[nm]).reshape(-1)
                )
        opt.zero_grad()
        loss.backward()
        opt.step()

    # Held-out exact match on the answer, teacher-forced one position at a time.
    m.eval()
    ok = 0
    with torch.no_grad():
        for j in range(0, len(Xte), 128):
            x, a = Xte[j : j + 128].to(device), Ate[j : j + 128].to(device)
            v = Vte[j : j + 128].to(device) if use_fone else None
            h, _ = m(x[:, :-1], torch.zeros(1), None, v[:, :-1] if use_fone else None)
            y = x[:, 1:]
            mask = a[:, :-1]
            if use_fone:
                nm = (y == num_id) & mask
                if nm.any():
                    w = Wte[j : j + 128].to(device)[:, 1:]
                    pred = fone.decode(m.num_logits(h[nm].float()))
                    ok += int((pred.cpu() == w[nm].double().cpu()).sum())
            else:
                pred = m.head(h)[..., : cfg.vocab].argmax(-1)
                # a problem counts only if every answer token is right
                right = ((pred == y) | ~mask).all(dim=1)
                ok += int(right.sum())
    total = len(Xte) if not use_fone else int(((Xte[:, 1:] == num_id) & Ate[:, :-1]).sum())
    return ok, total


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    print(f"two-digit +/- on held-out pairs, {a.steps} steps, device={a.device}\n")
    for flag in (False, True):
        ok, tot = run(flag, a.steps, a.device)
        name = "FoNE   " if flag else "BPE    "
        print(f"  {name} {ok}/{tot} = {100 * ok / max(tot, 1):.1f}%")
