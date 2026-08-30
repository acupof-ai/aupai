#!/usr/bin/env python3
"""Context-length probe for the NoPE ladder checkpoint (fb task 2026-08-31).

Architecture: seq=4096, NoPE (no RoPE, no learned position embeddings -- position is
carried by KDA state); the 1024 sliding window was deleted 2026-08-30; attention is
full-causal over 4096. 3 gated-MLA layers are mechanically length-free, 9 KDA layers
carry position in fixed-size recurrent state -- the two legs may diverge past training
length, and nobody has measured it. Agentic loops eat call+tool-output+continuation
per turn, so the usable context budget is a 30B design constraint.

Two readings, PRE-REGISTERED before running (2026-08-31):

1. Beyond 4096 (paired delta-NLL). Each held-out row pair (a, b) is scored twice:
   b fresh (4096 tokens) and b after a (8192). delta[p] = NLL(paired, pos 4096+p)
   - NLL(fresh b, pos p). Causal invariance makes this a pure context-length contrast:
   same tokens, same positions-in-row, only the carried context differs.
   READING: mean delta over the first 128 positions of b (absolute distance 4096-4224):
     > +0.10 nat  => HARD BOUNDARY: context past 4096 already hurts at the seam; the
                     agent loop must close within 4096 (hard 30B constraint).
     <= +0.10 nat and rising with distance => SMOOTH DEGRADATION: usable past 4096,
                     solvable with training length; report slope in nat/1024 tokens.
   (0.10 nat is ~10-20x the expected bin SE at n>=100 pairs; a boundary that small
   would be operationally irrelevant anyway.)

2. Within 4096 (per-position curve). Per-token NLL on fresh rows, 256-token bins.
   READING: effective budget = first position where bin NLL exceeds the min bin by
   > 0.10 nat, sustained to the end; if never, effective = 4096. A rising tail means
   the usable budget is shorter than the nominal 4096.

Data: the same held-out head as score_matrix domain_loss (domain_files + head_texts,
packed with EOS), so within-4096 numbers are comparable to the ladder val figure.

# restartable: per-batch NLL rows are appended (flushed) to --out JSONL as
# {"phase": "fresh"|"paired", "batch": i, "nll": [[...], ...]}; on rerun, completed
# (phase, batch) pairs are skipped, so an interrupt loses at most one batch
# (~15s of the ~15min run). The final summary JSON (--summary) is rewritten after
# every batch with the partial curves, so even the aggregate survives a kill.

Usage: CUDA_VISIBLE_DEVICES=7 python3 eval/ctx_probe.py --ckpt ckpt_p324.pt --mix data/mix_scale_3.24b.json
"""
import argparse
import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

# FLA_FLASH_KDA must stay unset: attn_every 4 ladder checkpoints route 9/12 layers
# through chunk_kda; the eval runners' "0" default makes that import fail.
from domain_loss import EOS_ID, HOLDOUT_ROWS, domain_files, head_texts  # noqa: E402
from scripts.loader import load_checkpoint, load_tokenizer  # noqa: E402

SEQ = 4096
LONG = 8192
BIN = 256
MAX_PAIRS = 256
DELTA_SEAM_NAT = 0.10  # pre-registered hard-boundary threshold


def pack_tokens(tok, texts):
    ids = []
    for t in texts:
        ids.extend(tok.encode(t).ids + [EOS_ID])
    return ids


def per_token_nll(model, x, bs):
    """Per-token next-token CE, [n, seq-1]. Plain CE (no softcap), same as domain_loss.
    no_grad is mandatory: at seq=8192 the autograd graph alone blows past 95GB."""
    outs = []
    for i in range(0, len(x), bs):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x[i : i + bs])
        if isinstance(logits, tuple):
            logits = logits[0]
        y = x[i : i + bs][:, 1:]
        nll = torch.nn.functional.cross_entropy(
            logits[:, :-1].float().reshape(-1, logits.shape[-1]),
            y.reshape(-1),
            reduction="none",
        ).view(y.shape)
        outs.append(nll.cpu())
    return torch.cat(outs, 0)


def bin_means(mat, bin_size):
    """mat [n, seq] -> bin means; a leftover tail (< bin_size) becomes its own bin."""
    n, seq = mat.shape
    full = (seq // bin_size) * bin_size
    bins = mat[:, :full].view(n, full // bin_size, bin_size).mean(dim=(0, 2))
    if full < seq:
        bins = torch.cat([bins, mat[:, full:].mean().unsqueeze(0)])
    return bins


def load_done(path):
    """Completed (phase, batch) pairs and their NLL rows, from the JSONL."""
    done = {}
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            done[(r["phase"], r["batch"])] = r["nll"]
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--mix", required=True)
    ap.add_argument("--tokenizer", default=os.path.join(ROOT, "data", "tokenizer.json"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "ctx_probe.jsonl"))
    ap.add_argument("--summary", default=os.path.join(ROOT, "runs", "ctx_probe_p324.json"))
    args = ap.parse_args()

    done = load_done(args.out)
    fout = open(args.out, "a", encoding="utf-8")

    def run_phase(model, x, phase, bs):
        """Per-token NLL with resume: skip batches already in the JSONL."""
        outs = [None] * ((len(x) + bs - 1) // bs)
        for (ph, b), nll in done.items():
            if ph == phase:
                outs[b] = torch.tensor(nll)
        for i in range(0, len(x), bs):
            b = i // bs
            if outs[b] is not None:
                continue
            nll = per_token_nll(model, x[i : i + bs], bs)
            outs[b] = nll
            fout.write(json.dumps({"phase": phase, "batch": b,
                                   "nll": nll.tolist()}) + "\n")
            fout.flush()
        return torch.cat(outs, 0)

    def write_summary(fresh_nll, paired_nll, n_pairs):
        within = bin_means(fresh_nll, BIN)
        mn, mni = within.min().item(), within.argmin().item()
        eff = SEQ
        for i in range(mni + 1, len(within)):
            if within[i] > mn + DELTA_SEAM_NAT and all(
                within[j] > mn + DELTA_SEAM_NAT for j in range(i, len(within))
            ):
                eff = i * BIN
                break
        out = {"n_pairs": n_pairs, "within_4096_bins": within.tolist(),
               "effective_budget": eff}
        if paired_nll is not None:
            delta = paired_nll[:, SEQ:] - fresh_nll[1::2][:n_pairs]
            delta_bins = bin_means(delta, BIN)
            win = torch.stack([paired_nll[:, a:b].mean() for a, b in
                               [(0, 2048), (2048, 4096), (4096, 6144), (6144, 8192)]])
            seam = delta_bins[0].item()
            tail = delta_bins[-1].item()
            slope = (delta_bins[-1] - delta_bins[0]).item() / 3.0
            if seam > DELTA_SEAM_NAT:
                r1 = (f"HARD BOUNDARY: seam delta {seam:+.4f} > +{DELTA_SEAM_NAT} nat "
                      f"-- context past 4096 hurts at the seam; agent loop must close within 4096")
            else:
                r1 = (f"SMOOTH DEGRADATION: seam delta {seam:+.4f} <= +{DELTA_SEAM_NAT} nat, "
                      f"slope {slope:+.4f} nat/1024tok, tail {tail:+.4f} "
                      f"-- usable past 4096, solvable with training length")
            out.update({"delta_bins": delta_bins.tolist(), "window_nll": win.tolist(),
                        "seam_delta": seam, "tail_delta": tail,
                        "slope_nat_per_1024": slope, "reading1": r1})
        with open(args.summary, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        if paired_nll is None:
            return within, None, None, None, eff, mn, mni
        return within, delta_bins, win, r1, eff, mn, mni

    model, cfg = load_checkpoint(args.ckpt, device=args.device, dtype=torch.bfloat16)
    tok = load_tokenizer(args.tokenizer, cfg)
    model.eval()

    files = domain_files(args.mix, ROOT)
    texts = []
    for name, p in files.items():
        texts.extend(head_texts(p, HOLDOUT_ROWS))
        print(f"loaded {name}: {p}", flush=True)
    ids = pack_tokens(tok, texts)
    n_rows = min((len(ids) - 1) // SEQ, MAX_PAIRS * 2)
    n_pairs = n_rows // 2
    print(f"{n_rows} rows = {n_pairs} pairs, {n_rows * SEQ / 1e6:.2f}M tokens", flush=True)

    rows = torch.tensor(ids[: n_rows * SEQ], dtype=torch.long).view(n_rows, SEQ).to(args.device)
    paired = torch.cat([rows[0::2][:n_pairs], rows[1::2][:n_pairs]], dim=1)

    fresh_nll = run_phase(model, rows, "fresh", args.bs)
    write_summary(fresh_nll, None, n_pairs)  # partial: within-4096 curve only
    paired_nll = run_phase(model, paired, "paired", args.bs)

    within, delta_bins, win, r1, eff, mn, mni = write_summary(fresh_nll, paired_nll, n_pairs)
    fout.close()

    spans = [(i * BIN, i * BIN + BIN - 1) for i in range(len(within) - 1)] + [((len(within) - 1) * BIN, 4095 - 1)]
    print("\n=== within-4096 per-position NLL (256-token bins, last = tail) ===")
    for (a, b), v in zip(spans, within.tolist()):
        print(f"  pos {a:5d}-{b:5d}: {v:.4f}")
    print("\n=== paired delta-NLL beyond 4096 (bins over positions-in-b) ===")
    for (a, b), v in zip([(4096 + x, 4096 + y) for x, y in spans], delta_bins.tolist()):
        print(f"  dist {a:5d}-{b:5d}: {v:+.4f}")
    print("\n=== mean NLL per 2048-window of the 8192 run ===")
    for i, v in enumerate(win.tolist()):
        print(f"  win {i*2048:5d}-{i*2048+2047:5d}: {v:.4f}")
    print(f"\nREADING 1: {r1}")
    print(f"READING 2: min bin {mn:.4f} at pos {mni*BIN}; effective budget = {eff}"
          f"{' (nominal, no sustained rise)' if eff == SEQ else ''}")
    print(f"\nsummary: {args.summary}\nrows: {args.out}")


if __name__ == "__main__":
    main()
