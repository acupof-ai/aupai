#!/usr/bin/env python3
# restartable: reads two checkpoints and the token cache; minutes, nothing to shard.
"""A/B (4) row-norm reading: did the VE table win, or did a few frequent rows win?

    python3 scripts/ve_row_norms.py --ckpt ckpt_ab_valueembed_valueembed.pt.ep1
    python3 scripts/ve_row_norms.py --selftest

PRE-REGISTERED IN THE EXP ROW, and this script exists because the two answers imply different
next steps: "the table bought something" argues for keeping a full [vocab, d] table at scale,
while "the 200 most frequent rows bought something" argues for a much smaller table or a
factorised one, at a fraction of the +16.3% parameters.

WHY ROW NORM ALONE CANNOT ANSWER "WHICH ROWS WERE TOUCHED". nn.Embedding initialises to
N(0, 1), so an UNTOUCHED row at d=1024 has expected norm sqrt(1024) = 32.0 -- not zero. A naive
"rows with norm > 0 were trained" reads 32832 of 32832 and says nothing. Two things separate
touched from untouched:

  1. The DELTA against a fresh table under the checkpoint's own seed. A row no gradient reached
     is bit-identical to its init; a row that was trained is not. This is exact, not a threshold.
  2. The token counts from the run's own data, which say how often each row COULD have been
     reached. Reported beside the norms so the reader sees frequency and movement together.

Both are needed: (1) alone says a row moved, (2) alone says a row was available. The claim the
exp row has to support is about the JOINT distribution -- whether the movement concentrates in
the frequent rows.
"""

import argparse
import json
import math
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

os.environ.setdefault("FLA_FLASH_KDA", "0")


def row_stats(trained, fresh):
    """Per-row L2 norm and the L2 of the delta from init. Both [vocab]."""
    norms = trained.norm(dim=1)
    delta = (trained - fresh).norm(dim=1)
    return norms, delta


def summarise(norms, delta, counts=None, topk=(10, 100, 1000)):
    """The reading: how many rows moved at all, and how concentrated the movement is."""
    v = norms.numel()
    moved = (delta > 0).sum().item()
    out = {
        "vocab_rows": v,
        "rows_moved": moved,
        "rows_moved_frac": moved / v,
        "rows_untouched": v - moved,
        # An untouched row keeps its init norm; reporting the two populations separately is the
        # whole point, since a mean over all rows is dominated by the untouched ones.
        "norm_mean_moved": norms[delta > 0].mean().item() if moved else None,
        "norm_mean_untouched": norms[delta == 0].mean().item() if moved < v else None,
        "delta_sum": delta.sum().item(),
    }
    order = torch.argsort(delta, descending=True)
    total = delta.sum().item()
    for k in topk:
        if k <= v and total > 0:
            out[f"delta_share_top{k}"] = delta[order[:k]].sum().item() / total
    if counts is not None:
        # Do the frequent rows carry the movement? Spearman-free and simple: the share of total
        # delta held by the k most FREQUENT rows, against the share held by the k that moved
        # most. If the two agree, frequency explains the movement.
        freq_order = torch.argsort(counts, descending=True)
        for k in topk:
            if k <= v and total > 0:
                out[f"delta_share_top{k}_by_frequency"] = \
                    delta[freq_order[:k]].sum().item() / total
        seen = (counts > 0).sum().item()
        out["rows_seen_in_data"] = seen
        out["rows_seen_frac"] = seen / v
        # The honest cross-check: a row that was SEEN must have moved, and a row that moved must
        # have been seen. Either violation means the counts and the checkpoint disagree.
        out["moved_but_unseen"] = int(((delta > 0) & (counts == 0)).sum().item())
        out["seen_but_unmoved"] = int(((delta == 0) & (counts > 0)).sum().item())
    return out


def _selftest():
    torch.manual_seed(0)
    v, d = 64, 8
    fresh = torch.randn(v, d)
    trained = fresh.clone()
    # Move exactly three rows, by very different amounts.
    trained[5] += 10.0
    trained[9] += 1.0
    trained[40] += 0.1
    counts = torch.zeros(v)
    counts[5] = 1000
    counts[9] = 10
    counts[40] = 1

    norms, delta = row_stats(trained, fresh)
    s = summarise(norms, delta, counts, topk=(1, 3))
    assert s["rows_moved"] == 3, s["rows_moved"]
    assert s["rows_untouched"] == v - 3
    assert s["moved_but_unseen"] == 0, s
    assert s["seen_but_unmoved"] == 0, s
    # The heaviest row must dominate: 10.0 of 11.1 total delta.
    assert 0.85 < s["delta_share_top1"] < 0.95, s["delta_share_top1"]
    assert abs(s["delta_share_top3"] - 1.0) < 1e-6, s["delta_share_top3"]
    # Frequency and movement agree here by construction, so both top1 shares match.
    assert abs(s["delta_share_top1"] - s["delta_share_top1_by_frequency"]) < 1e-6

    # THE CASE THAT MATTERS: frequency and movement DISAGREE. If the by-frequency share equalled
    # the by-delta share unconditionally, the comparison would be vacuous -- so build a world
    # where the most frequent row barely moved and check the two diverge.
    t2 = fresh.clone()
    t2[5] += 0.01      # most frequent, barely moves
    t2[40] += 10.0     # rare, moves a lot
    n2, d2 = row_stats(t2, fresh)
    s2 = summarise(n2, d2, counts, topk=(1,))
    assert s2["delta_share_top1"] > 0.99, s2["delta_share_top1"]
    assert s2["delta_share_top1_by_frequency"] < 0.01, s2["delta_share_top1_by_frequency"]

    # An UNTOUCHED table must read as zero rows moved, and its norms must be the init norms --
    # this is the check that stops "norm > 0" from being mistaken for "trained".
    n3, d3 = row_stats(fresh.clone(), fresh)
    s3 = summarise(n3, d3, counts, topk=(1,))
    assert s3["rows_moved"] == 0, s3
    assert s3["seen_but_unmoved"] == 3, s3
    assert s3["norm_mean_untouched"] is not None
    # And at the real width an untouched row's norm is ~sqrt(d), nowhere near zero.
    big = torch.randn(16, 1024)
    assert 0.9 * math.sqrt(1024) < big.norm(dim=1).mean().item() < 1.1 * math.sqrt(1024)

    # THE CONSTRUCTION-ORDER TRAP, asserted here because every check above hands `summarise` a
    # fresh table built by hand and so cannot see it. main() must reproduce the table the MODEL
    # made, and value_embed is NOT the first Embedding drawn from the RNG: model.py builds tok at
    # :326 and value_embed at :357. A bare `manual_seed(seed); nn.Embedding(v, d)` therefore
    # yields TOK's init, which differs from value_embed's on every row -- so the shortcut reports
    # a fully-moved table for an arm that never touched it. This asserts the two really do differ,
    # i.e. that the trap is live and the model-building path in main() is doing necessary work.
    v2, d2 = 512, 32
    torch.manual_seed(7)
    shortcut = torch.nn.Embedding(v2, d2).weight.detach().clone()
    torch.manual_seed(7)
    first = torch.nn.Embedding(v2, d2).weight.detach().clone()   # stands for tok
    second = torch.nn.Embedding(v2, d2).weight.detach().clone()  # stands for value_embed
    assert torch.equal(shortcut, first), \
        "the shortcut no longer reproduces the FIRST draw; this selftest's premise is stale"
    assert not torch.equal(shortcut, second), \
        "a bare nn.Embedding under the same seed reproduces the SECOND draw too, so the " \
        "construction-order trap is gone -- if torch changed this, main()'s model-building " \
        "path may be unnecessary, but verify before simplifying it"
    n4, d4 = row_stats(second, shortcut)
    assert (d4 > 0).sum().item() == v2, \
        f"the shortcut's table differs from the second draw on only {(d4 > 0).sum().item()} of " \
        f"{v2} rows; the measured claim is ALL rows"

    print("ve_row_norms selftest OK: counts only rows whose weights actually moved from init "
          "(an untouched row keeps norm ~sqrt(d)=32 at d=1024, so a norm threshold would call "
          "all 32832 rows trained), separates the moved and untouched populations, and the "
          "by-frequency share diverges from the by-delta share when frequency does NOT explain "
          "the movement, and the construction-order trap is live (a bare nn.Embedding under the "
          "seed reproduces the FIRST draw, i.e. tok, and differs from the second on ALL rows -- "
          "which is why main() builds the real model instead)")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", help="the VE arm checkpoint")
    ap.add_argument("--counts", help="optional json {token_id: count} from the run's data")
    ap.add_argument("--out", help="write the summary json here")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        return _selftest()
    if not a.ckpt:
        ap.error("--ckpt required (or --selftest)")

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    sd = ck["model"]
    key = next((k for k in sd if k.endswith("value_embed.weight")), None)
    if key is None:
        sys.exit(f"{a.ckpt} has no value_embed.weight: this is not a VE arm checkpoint. "
                 f"Keys present: {[k for k in list(sd)[:5]]}")
    trained = sd[key].float()

    # A FRESH table under the checkpoint's own seed and the model's own CONSTRUCTION ORDER.
    #
    # THE OBVIOUS VERSION IS WRONG AND I VERIFIED IT BEFORE USING IT: `manual_seed(seed);
    # nn.Embedding(v, d)` reproduces the FIRST embedding drawn from the RNG, which is `tok`
    # (model.py:326). `value_embed` is built at model.py:357, after tok and after every block, so
    # it consumes a different part of the stream. Measured: that shortcut's table is bit-equal to
    # tok's init and differs from value_embed's on 32832 of 32832 rows -- so the script would have
    # reported EVERY row moved on a table nobody touched, which is precisely the false reading
    # this analysis exists to avoid.
    #
    # So build the real model under the seed and read ITS value_embed. Costs one CPU model
    # construction and is exact.
    cfg = ck["cfg"]
    seed = cfg.get("seed", 42)
    import model as M  # noqa: PLC0415
    import train  # noqa: PLC0415

    M.chunk_kda = lambda q, k, v, **kw: (q * 0 + v, None)  # noqa: E731  no kernel needed for init
    M.HAS_FA = False
    c = train.Cfg
    for k, v in cfg.items():
        if hasattr(c, k) and not k.startswith("_"):
            setattr(c, k, v)
    c.value_embed = True
    torch.manual_seed(seed)
    fresh_model = M.HybridLM(c)
    if fresh_model.value_embed is None:
        sys.exit("the freshly built model has no value_embed table; cfg did not carry the flag")
    fresh = fresh_model.value_embed.weight.detach().float()
    if fresh.shape != trained.shape:
        sys.exit(f"fresh table is {tuple(fresh.shape)}, checkpoint's is {tuple(trained.shape)}; "
                 f"the cfg replay does not reproduce the arm's geometry")
    exact = torch.equal(trained, fresh)

    norms, delta = row_stats(trained, fresh)
    counts = None
    if a.counts:
        c = json.load(open(a.counts, encoding="utf-8"))
        counts = torch.zeros(trained.shape[0])
        for k, n in c.items():
            i = int(k)
            if i < counts.numel():
                counts[i] = float(n)

    s = summarise(norms, delta, counts)
    s["ckpt"] = a.ckpt
    s["seed_used_for_fresh"] = seed
    s["whole_table_identical_to_fresh"] = exact
    s["reading"] = ("rows_moved counts rows whose weights differ from a fresh table under the "
                    "same seed -- exact, not a threshold. An untouched row keeps its N(0,1) "
                    "init norm (~32 at d=1024), so a norm cutoff would call every row trained.")
    s["boundary"] = ("The fresh table is REPRODUCED by building the real model under the "
                     "checkpoint's seed, not read from a step-0 checkpoint, because value_embed "
                     "is not the first Embedding drawn from the RNG -- reproducing it with a "
                     "bare nn.Embedding gives tok's init instead and reports all 32832 rows "
                     "moved on an untouched table (measured). If the construction order in "
                     "model.py changes, this reading breaks the same way, and the tell is "
                     "rows_moved jumping to the full vocab. A run where "
                     "whole_table_identical_to_fresh is True for a TRAINED arm means the arm "
                     "never touched the table at all.")
    if a.out:
        json.dump(s, open(a.out, "w"), indent=1)
    print(json.dumps(s, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
