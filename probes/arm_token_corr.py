#!/usr/bin/env python3
"""Per-token loss correlation between two checkpoints on the SAME held-out tokens.

Answers one design question: is the paired-BPB estimator load-bearing? Pairing helps by
sqrt(2(1-corr)) / sqrt(2), so corr is the whole lever. corr 0.95 makes the paired SE 4.5x
tighter than unpaired; corr 0.5 makes it 1.4x and the design is not worth its complexity.

CPU-ONLY BY CONSTRUCTION. --device defaults to cpu and the script refuses cuda unless
--allow_cuda is passed, because the cards belong to another team by user order (2026-09-05)
and a probe that quietly takes one is the failure this repo has paid for twice.

THE VAL ROWS ARE READ THE WAY TRAINING READ THEM, not resampled: train.py:1989 takes
val = seqs[:n_val] per domain with n_val = min(max(1, int(len(seqs) * val_frac)),
val_rows_max), a deterministic prefix. So the tokens both arms validated on are recoverable
from the caches without rebuilding the mix -- which matters twice over: rebuilding would
read 35 GB per domain (the co-residency guard's population) and would also let a seed or
cache change silently substitute different tokens for the ones the arms actually scored.
mmap reads only the prefix.
"""
import argparse
import json
import math
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from scripts.loader import load_checkpoint, load_tokenizer  # noqa: E402

SEQ = 4096


def val_rows(domain, cache_path, val_frac, val_rows_max, seq=SEQ):
    """The exact val prefix train.py would have taken for this domain, via mmap.

    Returns (rows [n_val, seq+1] int64, n_rows_total). Reads the prefix only.
    """
    stream = torch.load(cache_path, map_location="cpu", weights_only=True, mmap=True)
    n_rows = stream.numel() // (seq + 1)
    n_val = min(max(1, int(n_rows * val_frac)), val_rows_max)
    flat = stream[: n_val * (seq + 1)]
    return flat.view(n_val, seq + 1).long(), n_rows, n_val


@torch.no_grad()
def token_losses(model, X, Y, batch, device):
    """Per-token cross-entropy, flattened. No reduction: the correlation is per token."""
    out = []
    for j in range(0, len(X), batch):
        xb = X[j : j + batch].to(device)
        yb = Y[j : j + batch].to(device)
        logits = model(xb)
        logits = logits[0] if isinstance(logits, tuple) else logits
        ls = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(), yb.reshape(-1), reduction="none"
        )
        out.append(ls.cpu())
        print(f"    rows {j + len(xb)}/{len(X)}", flush=True)
    return torch.cat(out)


def paired_stats(la, lb, rows_per_token):
    """Correlation, the difference SD, and what each buys for the paired design.

    rows_per_token maps each token to its document (row), because the SE of a mean over
    tokens is not sqrt(var/n) when tokens within a document are correlated -- the same
    clustering correction the api_cloze probe carries.
    """
    n = la.numel()
    ma, mb = la.mean().item(), lb.mean().item()
    sa, sb = la.std(unbiased=True).item(), lb.std(unbiased=True).item()
    cov = ((la - ma) * (lb - mb)).mean().item() * n / (n - 1)
    corr = cov / (sa * sb)
    d = la - lb
    sd_diff = d.std(unbiased=True).item()
    # Clustered SE of the mean difference: group by row, average within, then SE over rows.
    k = int(rows_per_token.max().item()) + 1
    sums = torch.zeros(k, dtype=torch.float64).index_add_(
        0, rows_per_token, d.to(torch.float64)
    )
    cnts = torch.zeros(k, dtype=torch.float64).index_add_(
        0, rows_per_token, torch.ones_like(d, dtype=torch.float64)
    )
    keep = cnts > 0
    per_row = (sums[keep] / cnts[keep])
    se_cluster = (per_row.std(unbiased=True) / math.sqrt(per_row.numel())).item()
    se_naive = sd_diff / math.sqrt(n)
    sd_indep = math.sqrt(sa * sa + sb * sb)
    # sd_diff == 0 means the two arms scored every token identically -- the same weights, or
    # the same checkpoint passed twice. Pairing's gain is then unbounded, and inf is the
    # honest value: it says "no difference to resolve", where a fallback of 1.0 would read as
    # "pairing buys nothing" and a NaN would propagate silently into a design decision.
    gain = sd_indep / sd_diff if sd_diff > 0 else float("inf")
    return {
        "n_tokens": n,
        "n_rows": int(keep.sum().item()),
        "mean_a": ma,
        "mean_b": mb,
        "mean_diff": d.mean().item(),
        "sd_a": sa,
        "sd_b": sb,
        "corr": corr,
        "sd_diff": sd_diff,
        "sd_diff_if_independent": sd_indep,
        "se_diff_naive": se_naive,
        "se_diff_cluster": se_cluster,
        "deff": (se_cluster / se_naive) ** 2 if se_naive > 0 else float("nan"),
        "pairing_gain_vs_unpaired": gain,
    }


def main():
    ap = argparse.ArgumentParser()
    # NOT required=True: --selftest must run without checkpoints, and argparse enforces
    # required arguments before any code sees the namespace, so a required --ckpt_a makes
    # `--selftest` exit 2 with a usage message rather than testing anything.
    ap.add_argument("--ckpt_a")
    ap.add_argument("--ckpt_b")
    ap.add_argument("--mix", default=os.path.join(ROOT, "data", "mix_200m_8b.json"))
    ap.add_argument("--domain", default="code_py_starcoder")
    ap.add_argument("--cache", default=None, help="override the domain cache path")
    ap.add_argument("--tokenizer", default=os.path.join(ROOT, "data", "tokenizer.json"))
    ap.add_argument("--rows", type=int, default=64, help="val rows to score (cost is linear)")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--val_rows_max", type=int, default=5000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--allow_cuda", action="store_true")
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        return _selftest()

    if not a.ckpt_a or not a.ckpt_b:
        sys.exit("--ckpt_a and --ckpt_b are both required to measure a correlation")

    if "cuda" in a.device and not a.allow_cuda:
        sys.exit(
            "REFUSING --device cuda without --allow_cuda: the cards are assigned to another "
            "team by user order (2026-09-05) and runs/card_assignment.json is the authority, "
            "not an idle nvidia-smi row. This probe is designed to run on CPU."
        )

    cache = a.cache or f"/data00/tokens_{a.domain}.pt"
    rows, n_rows_total, n_val = val_rows(a.domain, cache, a.val_frac, a.val_rows_max)
    take = min(a.rows, len(rows))
    rows = rows[:take]
    X, Y = rows[:, :-1].contiguous(), rows[:, 1:].contiguous()
    # Token -> row map for the clustered SE.
    rpt = torch.arange(take).repeat_interleave(SEQ)
    print(
        f"{a.domain}: cache holds {n_rows_total} rows, train's val prefix is {n_val}, "
        f"scoring {take} rows = {take * SEQ} tokens",
        flush=True,
    )

    losses = {}
    for tag, path in (("a", a.ckpt_a), ("b", a.ckpt_b)):
        print(f"  loading {path}", flush=True)
        model, cfg = load_checkpoint(path, device=a.device)
        load_tokenizer(a.tokenizer, cfg)  # cross-checks vocab_real then vocab_id
        model.eval()
        print(f"  scoring {tag}", flush=True)
        losses[tag] = token_losses(model, X, Y, a.batch, a.device)
        del model

    st = paired_stats(losses["a"], losses["b"], rpt)
    st.update(
        {
            "ckpt_a": a.ckpt_a,
            "ckpt_b": a.ckpt_b,
            "domain": a.domain,
            "rows_scored": take,
            "val_prefix_rows": n_val,
            "cache_rows": n_rows_total,
            "device": a.device,
        }
    )
    print(json.dumps(st, indent=1))
    print()
    print(f"corr = {st['corr']:.4f}")
    print(f"sd of the per-token difference = {st['sd_diff']:.4f} nat "
          f"(vs {st['sd_diff_if_independent']:.4f} if the arms were independent)")
    print(f"pairing buys {st['pairing_gain_vs_unpaired']:.2f}x on the SE")
    print(f"SE(mean diff): naive {st['se_diff_naive']:.5f}, "
          f"document-clustered {st['se_diff_cluster']:.5f}, deff {st['deff']:.1f}")
    print("The CLUSTERED SE is the one a design decision uses: tokens within a document are "
          "not independent, and the naive SE understates by sqrt(deff).")
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(st, fh)
    return 0


def _selftest():
    torch.manual_seed(0)
    n, k = 4096, 8
    rpt = torch.arange(k).repeat_interleave(n // k)

    # 1. IDENTICAL ARMS: corr 1, zero difference, and the gain is infinite rather than a
    #    quiet 1.0 -- a design whose two arms are the same model has no difference to
    #    measure, and reporting a finite gain there would hide that.
    x = torch.randn(n).abs()
    st = paired_stats(x, x.clone(), rpt)
    assert abs(st["corr"] - 1.0) < 1e-6, st["corr"]
    assert st["sd_diff"] == 0.0, st["sd_diff"]
    assert math.isinf(st["pairing_gain_vs_unpaired"]), st["pairing_gain_vs_unpaired"]

    # 2. INDEPENDENT ARMS: pairing buys nothing (gain 1.0), which is the null the design
    #    argument must beat. Same variance both sides so the algebra is checkable by hand.
    y = torch.randn(n)
    z = torch.randn(n)
    st = paired_stats(y, z, rpt)
    assert abs(st["corr"]) < 0.06, st["corr"]
    assert abs(st["pairing_gain_vs_unpaired"] - 1.0) < 0.06, st["pairing_gain_vs_unpaired"]

    # 3. THE GAIN IS sqrt(2/(2(1-corr))) AT EQUAL VARIANCE, checked against the closed form
    #    at a known correlation rather than asserted from the definition.
    g = torch.randn(n)
    for rho in (0.5, 0.9, 0.99):
        h = rho * g + math.sqrt(1 - rho * rho) * torch.randn(n)
        st = paired_stats(g, h, rpt)
        want = 1.0 / math.sqrt(2 * (1 - st["corr"]) / 2)
        assert abs(st["pairing_gain_vs_unpaired"] - want) / want < 0.02, (rho, st)

    # 4. CLUSTERING INFLATES THE SE OF THE DIFFERENCE. A difference that is constant within
    #    a document and varies across documents has deff ~ tokens-per-document; the naive SE
    #    would divide by 4096 tokens when the design really has 8 independent units.
    per_row = torch.tensor([0.5, -0.4, 0.3, -0.2, 0.1, -0.1, 0.2, -0.3])
    d = per_row.repeat_interleave(n // k)
    base = torch.randn(n)
    st = paired_stats(base + d, base, rpt)
    assert st["deff"] > 100, f"clustering not detected: deff {st['deff']}"
    # the clustered SE must be the SD of the 8 row means over sqrt(8)
    want = (per_row.std(unbiased=True) / math.sqrt(k)).item()
    assert abs(st["se_diff_cluster"] - want) < 1e-5, (st["se_diff_cluster"], want)

    # 5. NO CLUSTERING -> deff ~ 1. The correction must not invent a design effect where
    #    the data has none, or every SE it reports is inflated.
    st = paired_stats(torch.randn(n), torch.randn(n), rpt)
    assert 0.5 < st["deff"] < 2.0, st["deff"]

    # 6. THE CUDA REFUSAL IS IN main() AND READS --allow_cuda. Checked by source, because
    #    calling main() needs two checkpoints on disk. The cards belong to another team;
    #    a probe that takes one silently is the defect, not the wrong number.
    import inspect

    src = inspect.getsource(main)
    assert 'if "cuda" in a.device and not a.allow_cuda:' in src, (
        "the CPU-only guard is gone or its condition changed"
    )
    assert "REFUSING --device cuda" in src

    # 7. val_rows TAKES A PREFIX AND CAPS IT, matching train.py:1989. Off-by-one here would
    #    score tokens the arms never validated on while still printing a correlation.
    class _Stream:
        def __init__(self, n_rows):
            self.t = torch.arange(n_rows * (SEQ + 1), dtype=torch.int32)

    real_load = torch.load
    try:
        torch.load = lambda *_, **__: _Stream(1000).t
        rows, total, n_val = val_rows("x", "unused", 0.05, 5000)
        assert total == 1000 and n_val == 50, (total, n_val)
        assert rows.shape == (50, SEQ + 1), rows.shape
        assert rows[0, 0].item() == 0 and rows[1, 0].item() == SEQ + 1
        # the cap binds before the fraction on a large domain
        torch.load = lambda *_, **__: _Stream(200000).t
        _, total, n_val = val_rows("x", "unused", 0.05, 5000)
        assert total == 200000 and n_val == 5000, (total, n_val)
        # and at least one row is kept on a tiny domain
        torch.load = lambda *_, **__: _Stream(3).t
        _, total, n_val = val_rows("x", "unused", 0.05, 5000)
        assert n_val == 1, n_val
    finally:
        torch.load = real_load

    print(
        "arm_corr selftest OK: identical arms give corr 1 and an infinite gain, independent "
        "arms give gain 1.00, the gain matches 1/sqrt(1-corr) at rho 0.5/0.9/0.99, a "
        "within-document difference is caught as deff>100 with the clustered SE equal to the "
        "row means' SE over sqrt(k), unclustered data keeps deff ~1, the val prefix matches "
        "train.py:1989 on three domain sizes (fraction, cap, and the 1-row floor), and the "
        "CPU-only refusal is present in main()"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
