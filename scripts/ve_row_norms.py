#!/usr/bin/env python3
# restartable: reads two checkpoints and the token cache; minutes, nothing to shard.
"""A/B (4) row-norm reading: did the VE table win, or did a few frequent rows win?

    python3 scripts/ve_row_norms.py --ckpt ckpt_ab_valueembed_valueembed.pt.ep1
    python3 scripts/ve_row_norms.py --selftest

PRE-REGISTERED IN THE EXP ROW, and this script exists because the two answers imply different
next steps: "the table bought something" argues for keeping a full [vocab, d] table at scale,
while "the 200 most frequent rows bought something" argues for a much smaller table or a
factorised one, at a fraction of the +16.3% parameters.

TWO READINGS THAT LOOK RIGHT AND ARE NOT. Both were caught before the number was used, and
both would have produced a confident answer to the pre-registered question:

  1. "Rows with norm > 0 were trained" reads 32832 of 32832. nn.Embedding inits to N(0, 1), so an
     untouched row at d=1024 has norm ~sqrt(1024) = 32, not zero.

  2. "Rows whose weights differ from init were trained" ALSO reads 32832 of 32832 -- and this is
     the one I was about to ship. The table sits in the AdamW `embed` group with
     weight_decay=0.001 (train.py:779), an Embedding's gradient is a DENSE zero-filled tensor,
     and AdamW's decoupled decay therefore steps every row whether or not a token reached it.
     Measured on a toy: after one step all 10 of 10 rows moved, touched rows by 2.0e-2 and
     untouched rows by 1.3e-5.

So neither presence nor movement separates the populations; MAGNITUDE does, and the two are three
orders apart. The reading is the delta distribution against a fresh table, with the decay floor
estimated from the data (the median row, since most rows are untouched at 0.26B tokens) rather
than assumed -- and the concentration of that delta, which answers the pre-registered question
without needing token frequencies at all: if a few hundred rows of 32832 hold nearly all the
gradient movement, the TABLE is not what won, whatever the frequency ranking was.

Token counts remain optional (--counts) and add only the frequency ORDERING, i.e. whether the
rows that moved are the frequent ones. They are not required for the verdict, and scanning the
mix's caches for them costs 35 GB of reads on one domain alone.
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
    """The reading: how the movement is DISTRIBUTED across rows.

    `rows_moved` is deliberately NOT the headline, and the reason is measured: the VE table sits
    in the AdamW `embed` group with weight_decay=0.001 (train.py:779), an Embedding's gradient is
    a DENSE zero-filled tensor, and AdamW's decoupled decay therefore steps EVERY row whether a
    token reached it or not. Verified on a toy: after one step, all 10 of 10 rows have moved --
    touched rows by 2.0e-2 and untouched rows by 1.3e-5, three orders apart. So a moved/unmoved
    count reads 32832 of 32832 and says nothing, exactly like the norm threshold it replaced.

    What separates the populations is MAGNITUDE. A decay-only row moves by about lr*wd*|p| per
    step, uniformly and tiny; a row that received gradient moves by its gradient. The reading is
    therefore the delta distribution and its concentration, with a decay floor estimated from the
    data itself (the median, since most rows are untouched at 0.26B tokens) rather than assumed.
    """
    v = norms.numel()
    moved = (delta > 0).sum().item()
    med = delta.median().item()
    # The decay floor: most rows are untouched at this token count, so the MEDIAN row is a
    # decay-only row. 10x that is a generous line between "decay only" and "gradient reached it".
    floor = max(med * 10, 0.0)
    above = (delta > floor).sum().item()
    out = {
        "vocab_rows": v,
        "rows_moved_at_all": moved,
        "delta_median": med,
        "decay_floor_10x_median": floor,
        "rows_above_decay_floor": above,
        "rows_above_decay_floor_frac": above / v,
        "delta_sum": delta.sum().item(),
        "norm_mean_above_floor": norms[delta > floor].mean().item() if above else None,
        "norm_mean_below_floor": norms[delta <= floor].mean().item() if above < v else None,
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
        # Cross-check against the floor, not against "moved": every row moves under decay.
        out["above_floor_but_unseen"] = int(((delta > floor) & (counts == 0)).sum().item())
        out["seen_but_below_floor"] = int(((delta <= floor) & (counts > 0)).sum().item())
    return out


def _selftest():
    torch.manual_seed(0)
    v, d = 64, 8
    fresh = torch.randn(v, d)

    # THE DECAY WORLD, which is the real one: AdamW's decoupled decay steps EVERY row because an
    # Embedding's grad is dense. So build a table where all 64 rows moved a little and three moved
    # a lot, and require the reading to separate them. A moved/unmoved count cannot.
    trained = fresh * (1 - 1e-5)          # decay floor on every row
    trained[5] += 10.0
    trained[9] += 1.0
    trained[40] += 0.1
    counts = torch.zeros(v)
    counts[5], counts[9], counts[40] = 1000, 10, 1

    norms, delta = row_stats(trained, fresh)
    s = summarise(norms, delta, counts, topk=(1, 3))
    assert s["rows_moved_at_all"] == v, \
        f"only {s['rows_moved_at_all']} of {v} rows moved; the decay fixture is not realistic"
    assert s["rows_above_decay_floor"] == 3, \
        (f"the floor admits {s['rows_above_decay_floor']} rows, expected the 3 with gradient. "
         f"A moved/unmoved count would have said {v}.")
    assert s["above_floor_but_unseen"] == 0, s
    assert s["seen_but_below_floor"] == 0, s
    # The heaviest row must dominate: 10.0 of ~11.1 total gradient movement.
    assert 0.80 < s["delta_share_top1"] < 0.95, s["delta_share_top1"]
    # Frequency and movement agree here by construction.
    assert abs(s["delta_share_top1"] - s["delta_share_top1_by_frequency"]) < 1e-6

    # THE CASE THAT MATTERS: frequency and movement DISAGREE. If the by-frequency share equalled
    # the by-delta share unconditionally, the comparison would be vacuous -- so build a world
    # where the most frequent row barely moved and check the two diverge.
    t2 = fresh * (1 - 1e-5)
    t2[5] += 0.01      # most frequent, barely moves
    t2[40] += 10.0     # rare, moves a lot
    n2, d2 = row_stats(t2, fresh)
    s2 = summarise(n2, d2, counts, topk=(1,))
    assert s2["delta_share_top1"] > 0.90, s2["delta_share_top1"]
    assert s2["delta_share_top1_by_frequency"] < 0.05, s2["delta_share_top1_by_frequency"]

    # A DECAY-ONLY table -- no row received gradient -- must read as ZERO rows above the floor,
    # even though every row moved. This is the check that stops the decay floor from being
    # mistaken for training, and it is the shape the real 0.26B run may well have.
    t3 = fresh * (1 - 1e-5)
    n3, d3 = row_stats(t3, fresh)
    s3 = summarise(n3, d3, counts, topk=(1,))
    assert s3["rows_moved_at_all"] == v, s3["rows_moved_at_all"]
    assert s3["rows_above_decay_floor"] == 0, \
        (f"{s3['rows_above_decay_floor']} rows read as trained in a table where NOTHING but "
         f"decay happened; the floor is not separating the populations")
    assert s3["seen_but_below_floor"] == 3, s3

    # And at the real width an untouched row's norm is ~sqrt(d), nowhere near zero -- so no norm
    # threshold can separate the populations either.
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

    print("ve_row_norms selftest OK: separates gradient movement from the AdamW decay floor "
          "(decay steps EVERY row, so a moved/unmoved count reads the whole vocab -- measured "
          "2.0e-2 touched against 1.3e-5 decay-only), reads zero trained rows in a decay-only "
          "table, no norm threshold could do either since an untouched row keeps norm "
          "~sqrt(d)=32 at d=1024, and the "
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
