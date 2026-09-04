#!/usr/bin/env python3
# restartable: forward-only scoring of two checkpoints over one val subset, writing one small JSON at
# the end. An interrupt costs the run (minutes of card time) and nothing else -- no partial file a
# later run could mistake for complete, no cache rewritten (val_seqs asserts freshness, never rebuilds).
"""Is the N8 fix's corpus gain leak removal, or a row-edge effect the mask also produces?

THE QUESTION, from b0's A/B (fixed vs current, only conv_doc_isolated differing): corpus loss
-0.024353 nat, 552/576 blocks, 9/9 domains -- but the gain does NOT track documents per row
(corr -0.56, against -0.95 for the cu eval artifact). zh_web at ~2 eos/row gains -0.036 and chatml at
~18 gains -0.037. If the gain were removal of cross-document leakage it would scale with the number of
boundaries; it does not.

THE CANDIDATE (6e): the mask also zeroes the conv taps at the first K-1 = 3 positions of every ROW,
which is domain-independent -- every row has exactly one row start no matter how many documents it
holds. Both hypotheses act on the same POSITIONS, since a row start is also a document start, so
position cannot separate them. What separates them is COUNT: leak removal is a function of INTERIOR
document boundaries (boundaries minus the row start), the row-edge effect is not.

WHY THIS DESIGN AND NOT THE OBVIOUS ONE. Comparing positions 0-2 against 3+ inside the fixed
checkpoint measures the CONTEXT RAMP: position 0 has one token of history and position 500 has five
hundred, and per-token loss falls steeply over the first tokens of any causal model. Every model shows
that gap, with or without the flag. So the contrast has to be fixed-vs-current at the SAME positions.

THE DISCRIMINATOR IS SINGLE-DOCUMENT ROWS. On a row holding exactly one document there are ZERO
interior boundaries, so:

    leak removal      predicts NO gain at positions 0-2 (there is nothing to leak from)
    row-edge effect   predicts the SAME gain as anywhere else (the row start is still masked)

That is the only place the two hypotheses predict different numbers rather than the same number under
different labels. Hence 6e's ruling: run contrast (b) alone.

THE CONTROL COMES FIRST, AND ITS NUMBER IS REPORTED FIRST. The effect under test is -0.024 nat over a
corpus, and bf16 reduction order moves per-token values by more than that at this granularity: the
T-parity measurement (runs/n8/) found appending ONE token to a row shifts earlier positions by 2-4
absolute, mean 6.7 ulps, purely from chunk_kda's tiling changing with T.

THE FIRST VERSION'S CONTROL WAS VACUOUS AND MEASURED 0.000000. It scored each arm TWICE on the same
rows and called the difference the resolution -- but one checkpoint over the same rows at the same
shapes is BITWISE deterministic, so it measured determinism and produced a floor of nothing, against
which any gap "exceeds the resolution". Reduction order is fixed by the shapes, and the shapes were
deliberately held fixed. The control now perturbs the one thing that changes the arithmetic without
changing the quantity: row LENGTH. The same rows are right-padded with <eos> to a second length, the
edge positions sit at the row START so the measured tokens are untouched, and the same-arm difference
across the two lengths is the floor. A gap inside that floor is "cannot see", never "no effect".

THE POOLED NUMBER IS A FOOTNOTE, THE PER-DOMAIN TABLE IS THE RESULT. Single-document rows exist only
where rows hold few documents, so the subset is not the corpus: measured on the first run, chat_qa
contributed 0 rows and code_py_rp1t contributed 23 of 53. A pooled mean over that is a number about two
code domains wearing a corpus label. Each domain's gap is printed against its OWN resolution.

    python3 scripts/e1_n8_row_edge_probe.py --fixed <ckpt> --current <ckpt>
    python3 scripts/e1_n8_row_edge_probe.py --selftest
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "eval"))

#: Positions after a boundary that the k=4 conv can contaminate. K-1 = 3 taps reach backwards, so
#: offsets 0,1,2 are the affected ones and 3+ are exactly 0.0000 (measured, model.py:119).
EDGE = 3

#: The minimum single-document val rows worth reporting. Below this the per-position mean over three
#: offsets has no resolution against the bf16 floor, and a number computed from a handful of rows
#: would be quoted as if it settled the question. Refuse instead of reporting.
MIN_ROWS = 32


def single_doc_rows(seqs, eos_id):
    """Indices of rows holding exactly ONE document: no interior <eos> before the last position.

    A trailing <eos> is the document's own terminator, not an interior boundary, so it is excluded
    from the count -- a row that ends with <eos> and has none inside still holds one document. Getting
    that wrong would empty the subset (if every packed row ends with <eos>) or fill it with two-doc
    rows, and both would look like a working probe.
    """
    out = []
    for i in range(seqs.shape[0]):
        row = seqs[i]
        interior = int((row[:-1] == eos_id).sum())
        if interior == 0:
            out.append(i)
    return out


def per_position_ce(model, x, y, offsets, cu_of=None):
    """Mean CE at the given row offsets, and at every other position, for one batch of rows.

    Returns (edge_sum, edge_tokens, rest_sum, rest_tokens). The "rest" half is not the contrast -- it
    is the sanity number that says whether the two arms differ everywhere or only at the edge.
    """
    import torch

    edge_s = rest_s = 0.0
    edge_n = rest_n = 0
    for i in range(x.shape[0]):
        xb = x[i : i + 1]
        yb = y[i : i + 1]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=xb.is_cuda):
            logits = model(xb, cu=cu_of(xb)) if cu_of else model(xb)
        if isinstance(logits, tuple):
            logits = logits[0]
        flat = logits.float().view(-1, logits.shape[-1])
        per = torch.nn.functional.cross_entropy(
            flat, yb.reshape(-1), reduction="none").view(yb.shape)[0]
        mask = torch.zeros_like(per, dtype=torch.bool)
        for o in offsets:
            if o < mask.shape[0]:
                mask[o] = True
        edge_s += float(per[mask].sum())
        edge_n += int(mask.sum())
        rest_s += float(per[~mask].sum())
        rest_n += int((~mask).sum())
    return edge_s, edge_n, rest_s, rest_n


def selftest():
    """Known answers for the two pieces that decide whether the probe measures anything."""
    import torch

    bad = 0
    eos = 7
    # A row with an interior eos is TWO documents; a row whose only eos is the terminator is ONE.
    seqs = torch.tensor([
        [1, 2, 3, 4, 5, 6],        # no eos at all -> one document
        [1, 2, 3, 4, 5, eos],      # trailing eos only -> still one document
        [1, 2, eos, 4, 5, 6],      # interior eos -> two documents
        [eos, 2, 3, 4, 5, eos],    # eos at position 0 is interior -> two documents
    ])
    got = single_doc_rows(seqs, eos)
    if got != [0, 1]:
        print(f"  FAIL single_doc_rows returned {got}, expected [0, 1]: a trailing eos is the "
              f"document's terminator, an interior one is a boundary")
        bad += 1
    # THE SUBSET MUST NOT BE EMPTY OR UNIVERSAL on a realistic mix, or the probe reports a number
    # from no rows / from every row and the discriminator is gone either way.
    if not got:
        print("  FAIL the single-document subset is empty; the contrast cannot be computed")
        bad += 1
    if len(got) == seqs.shape[0]:
        print("  FAIL every row was called single-document, so interior boundaries are not detected")
        bad += 1
    # EDGE is the conv's reach, not an arbitrary window: K-1 for k=4.
    if EDGE != 3:
        print(f"  FAIL EDGE is {EDGE}; the k=4 conv reaches 3 positions back (model.py:113)")
        bad += 1
    # per_position_ce's masking, on a fixture with a known answer: a constant-logit model gives the
    # same CE at every position, so edge and rest means must be EQUAL and the counts must partition.
    V, T = 5, 8

    class Const(torch.nn.Module):
        def forward(self, xb, cu=None):
            return torch.zeros(xb.shape[0], xb.shape[1], V)

    x = torch.zeros(4, T, dtype=torch.long)
    y = torch.zeros(4, T, dtype=torch.long)
    es, en, rs, rn = per_position_ce(Const(), x, y, range(EDGE))
    if en != 4 * EDGE:
        print(f"  FAIL edge token count {en}, expected {4 * EDGE} (3 offsets x 4 rows)")
        bad += 1
    if en + rn != 4 * T:
        print(f"  FAIL edge+rest tokens {en + rn} != {4 * T}: the masks do not partition the row")
        bad += 1
    if abs(es / en - rs / rn) > 1e-5:
        print(f"  FAIL a constant-logit model gave different edge/rest means "
              f"({es / en:.6f} vs {rs / rn:.6f}); the masking is selecting different text")
        bad += 1
    # AND THE FIXTURE MUST HAVE DISCRIMINATING POWER: a constant-logit model cannot tell a correct
    # mask from a wrong one, so the case above is necessary and not sufficient. This one is the
    # sufficient half -- a position-dependent loss must give DIFFERENT edge and rest means.
    class Ramp(torch.nn.Module):
        def forward(self, xb, cu=None):
            out = torch.zeros(xb.shape[0], xb.shape[1], V)
            for t in range(xb.shape[1]):
                out[:, t, 0] = float(t)      # later positions grow confident in class 0
            return out

    es2, en2, rs2, rn2 = per_position_ce(Ramp(), x, y, range(EDGE))
    if abs(es2 / en2 - rs2 / rn2) < 1e-3:
        print("  FAIL a position-dependent loss gave equal edge/rest means, so the offsets are not "
              "selecting the positions they name -- the constant-logit case cannot catch this")
        bad += 1
    if es2 / en2 <= rs2 / rn2:
        print(f"  FAIL edge mean {es2 / en2:.4f} is not above rest mean {rs2 / rn2:.4f}; early "
              f"positions must be the WORSE ones under a confidence ramp")
        bad += 1
    print(f"e1 n8 row-edge probe selftest: {'OK' if not bad else f'{bad} FAILURE(S)'}")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixed", default="ckpt_b0_n8_fixed.pt")
    # ckpt_b0_sd_unlooped.pt, NOT ckpt_b0_sd_equalcompute.pt. My first run used equalcompute and the
    # fixed arm lost 0.02-0.04 everywhere -- because equalcompute is the 4824-step / 1.2646B-token arm
    # from the loop question, a better model by construction, against fixed's 3815 steps / 1.0001B.
    # 26% more tokens is what that gap was. unlooped is the arm that differs from fixed ONLY in the
    # flag: same steps, tokens, seed and mix (6e, 2026-09-04).
    ap.add_argument("--current", default="ckpt_b0_sd_unlooped.pt")
    ap.add_argument("--domains", nargs="*", default=None,
                    help="default: every domain in data/mix_200m_4b.json")
    ap.add_argument("--pad", type=int, default=64,
                    help="right-pad width for the T-parity resolution control")
    ap.add_argument("--out", default="runs/e1_n8_row_edge.json")
    ap.add_argument("--tokenizer", default=os.path.join(ROOT, "data", "tokenizer.json"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    import torch
    from cache_guard import set_vocab_id
    from domain_loss import EOS_ID, val_seqs

    from scripts.loader import load_checkpoint, load_tokenizer
    from train import doc_cu_seqlens

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    with open(os.path.join(ROOT, "data", "mix_200m_4b.json"), encoding="utf-8") as fh:
        domains = a.domains or list(json.load(fh)["domains"])

    # 1. THE SUBSET, built once and shared by both arms. Identical rows in identical order at
    #    identical lengths: the T-parity finding is that row LENGTH alone changes earlier positions
    #    by 6.7 ulps, so an arm scored on a different subset is not comparable at this resolution.
    #
    #    The TOKENIZER comes from the FIXED arm's cfg via load_tokenizer, which cross-checks vocab_id
    #    and raises on a mismatch. Both arms must share it: val_seqs reads the token cache, and a
    #    cache tokenized under another vocabulary would be a different subset with the same name.
    fixed_model, fixed_cfg = load_checkpoint(os.path.join(ROOT, a.fixed), device=dev,
                                            dtype=torch.bfloat16)
    set_vocab_id(fixed_cfg)
    tok = load_tokenizer(a.tokenizer, fixed_cfg)
    del fixed_model
    torch.cuda.empty_cache() if dev == "cuda" else None
    subset = {}
    for dom in domains:
        seqs = val_seqs(dom, tok)
        if seqs is None:
            print(f"  {dom}: no shards -- SKIPPED (absent, not zero)", flush=True)
            continue
        idx = single_doc_rows(seqs, EOS_ID)
        if idx:
            subset[dom] = seqs[idx]
        print(f"  {dom:22s} {len(idx):5d} single-document of {seqs.shape[0]} val rows", flush=True)
    n_rows = sum(v.shape[0] for v in subset.values())
    if n_rows < MIN_ROWS:
        raise SystemExit(
            f"REFUSING: {n_rows} single-document val rows across {len(subset)} domain(s), under the "
            f"{MIN_ROWS} this probe needs. The contrast is a mean over {EDGE} positions per row; "
            f"from {n_rows} rows it would be quoted as settling the leak-vs-row-edge question while "
            f"resolving nothing. Widen the val cap or report the subset as too small -- do not "
            f"lower MIN_ROWS to make this run.")
    print(f"subset: {n_rows} single-document rows over {len(subset)} domain(s)")

    # 2. SCORE. Each arm TWICE -- the same-arm difference is the resolution floor, and it is
    #    computed before any cross-arm number is looked at.
    def score(ckpt):
        mdl, _cfg = load_checkpoint(os.path.join(ROOT, ckpt), device=dev, dtype=torch.bfloat16)
        mdl.eval()
        acc = {}
        with torch.no_grad():
            for dom, seqs in subset.items():
                s = seqs.to(dev)
                x, y = s[:, :-1], s[:, 1:]
                acc[dom] = per_position_ce(mdl, x, y, range(EDGE),
                                           cu_of=lambda b: doc_cu_seqlens(b, EOS_ID))
        del mdl
        if dev == "cuda":
            torch.cuda.empty_cache()
        return acc

    # THE RESOLUTION CONTROL IS A T-PARITY PERTURBATION, not a repeat run.
    #
    # MY FIRST VERSION SCORED EACH ARM TWICE ON THE SAME ROWS and reported the difference as the
    # resolution: it came back 0.000000 for both arms, because the same checkpoint over the same rows
    # in the same order at the same shapes is BITWISE deterministic. That measured determinism, which
    # was never in question, and gave a floor of zero -- so "the gap exceeds the resolution" compared
    # the gap against nothing. The whole point of the control was the bf16 reduction-order noise, and
    # reduction order is fixed by the shapes, which I had deliberately held fixed.
    #
    # What moves reduction order while leaving the QUANTITY alone is the row LENGTH: chunk_kda's
    # tiling changes with T, which is where the 6.7-ulp T-parity figure came from. So the control
    # scores the same rows PADDED to a second length and takes the same-arm difference across the two
    # lengths. The measured quantity is unchanged (the same tokens, at the same offsets, under the same
    # weights); only the arithmetic path differs.
    def score_padded(ckpt, pad):
        """Same rows, padded on the RIGHT with <eos> to a longer T. The edge positions are at the row
        START, so right-padding cannot touch the tokens being measured -- it only changes the tiling.
        """
        mdl, _cfg = load_checkpoint(os.path.join(ROOT, ckpt), device=dev, dtype=torch.bfloat16)
        mdl.eval()
        acc = {}
        with torch.no_grad():
            for dom, seqs in subset.items():
                s = seqs.to(dev)
                s = torch.cat([s, torch.full((s.shape[0], pad), EOS_ID, dtype=s.dtype,
                                             device=s.device)], dim=1)
                x, y = s[:, :-1], s[:, 1:]
                acc[dom] = per_position_ce(mdl, x, y, range(EDGE),
                                           cu_of=lambda b: doc_cu_seqlens(b, EOS_ID))
        del mdl
        if dev == "cuda":
            torch.cuda.empty_cache()
        return acc

    runs = {}
    for name, ckpt in (("fixed", a.fixed), ("current", a.current)):
        runs[name] = score(ckpt)
        runs[name + "_padded"] = score_padded(ckpt, a.pad)
        for suffix in ("", "_padded"):
            r = runs[name + suffix]
            tot = sum(v[0] for v in r.values()), sum(v[1] for v in r.values())
            print(f"  {name + suffix:16s} edge mean {tot[0] / tot[1]:.6f} over {tot[1]} tokens",
                  flush=True)

    def edge_mean(r):
        s = sum(v[0] for v in r.values())
        n = sum(v[1] for v in r.values())
        return s / n

    def rest_mean(r):
        s = sum(v[2] for v in r.values())
        n = sum(v[3] for v in r.values())
        return s / n

    res_fixed = abs(edge_mean(runs["fixed"]) - edge_mean(runs["fixed_padded"]))
    res_current = abs(edge_mean(runs["current"]) - edge_mean(runs["current_padded"]))
    resolution = max(res_fixed, res_current)
    gap = edge_mean(runs["fixed"]) - edge_mean(runs["current"])
    gap_rest = rest_mean(runs["fixed"]) - rest_mean(runs["current"])
    # PER DOMAIN, because the pooled subset is NOT the corpus: single-document rows exist only where
    # rows hold few documents, so chat_qa contributed 0 and chatml near 0 while code_py_rp1t
    # contributed 23 of 53 on the first run. Pooling over that is a number about two code domains
    # wearing a corpus label. The token count travels with each domain's gap for the same reason.
    per_domain = {}
    for dom in subset:
        f_s, f_n = runs["fixed"][dom][0], runs["fixed"][dom][1]
        c_s, c_n = runs["current"][dom][0], runs["current"][dom][1]
        fp_s, fp_n = runs["fixed_padded"][dom][0], runs["fixed_padded"][dom][1]
        per_domain[dom] = {
            "edge_tokens": f_n, "rows": int(subset[dom].shape[0]),
            "fixed": f_s / f_n, "current": c_s / c_n, "gap": f_s / f_n - c_s / c_n,
            "resolution": abs(f_s / f_n - fp_s / fp_n),
        }

    out = {
        "question": ("is the N8 fix's corpus gain leak removal or a row-edge effect? On "
                     "single-document rows there are ZERO interior boundaries, so leak removal "
                     "predicts no gain at row positions 0-2 and the row-edge effect predicts the "
                     "same gain as anywhere"),
        "arms": {"fixed": a.fixed, "current": a.current},
        "rows": n_rows, "domains": sorted(subset), "edge_positions": list(range(EDGE)),
        "resolution_method": (f"T-parity: the same rows right-padded with <eos> by {a.pad}, same arm, "
                              f"so the measured tokens are identical and only chunk_kda's tiling "
                              f"changes. NOT a repeat run -- scoring one checkpoint twice at the same "
                              f"shapes is bitwise deterministic and reported 0.000000, a floor of "
                              f"nothing."),
        "resolution": resolution,
        "resolution_fixed": res_fixed, "resolution_current": res_current,
        "edge_gap_fixed_minus_current": gap,
        "rest_gap_fixed_minus_current": gap_rest,
        "edge_mean": {k: edge_mean(v) for k, v in runs.items()},
        "rest_mean": {k: rest_mean(v) for k, v in runs.items()},
        "per_domain": per_domain,
        "subset_is_not_the_corpus": ("single-document rows exist only where rows hold few documents, "
                                     "so the pooled number is dominated by those domains and the "
                                     "per-domain table is the result; pooled is a footnote"),
        "reading": None,
    }
    # THE RESOLUTION IS READ FIRST, and the verdict is written by the code rather than by whoever
    # reads the numbers: a gap inside the floor is "cannot see", not "no effect" (6e's ruling).
    n_over = sum(1 for v in per_domain.values() if abs(v["gap"]) > max(v["resolution"], 1e-12))
    if abs(gap) <= resolution:
        out["reading"] = (f"CANNOT SEE: the pooled edge gap {gap:+.6f} is inside the T-parity "
                          f"resolution {resolution:.6f}. This does not distinguish the hypotheses "
                          f"and is not evidence of no effect.")
    else:
        out["reading"] = (f"the pooled edge gap {gap:+.6f} exceeds the {resolution:.6f} resolution, "
                          f"and {n_over} of {len(per_domain)} domains exceed their own. On rows with "
                          f"ZERO interior boundaries leak removal predicts no gain, so a gain here is "
                          f"the ROW-EDGE effect; rest-of-row gap {gap_rest:+.6f}. Read the per-domain "
                          f"table before the pooled number: the subset is not the corpus.")
    dest = os.path.join(ROOT, a.out)
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nRESOLUTION (T-parity, same arm at two lengths): {resolution:.6f} nat/token "
          f"[fixed {res_fixed:.6f}, current {res_current:.6f}]")
    print(f"\n{'domain':22s}{'rows':>6s}{'edge tok':>10s}{'gap':>12s}{'own res':>12s}  verdict")
    for dom, v in sorted(per_domain.items(), key=lambda kv: -abs(kv[1]["gap"])):
        seen = abs(v["gap"]) > max(v["resolution"], 1e-12)
        print(f"{dom:22s}{v['rows']:6d}{v['edge_tokens']:10d}{v['gap']:+12.6f}"
              f"{v['resolution']:12.6f}  {'above floor' if seen else 'CANNOT SEE'}")
    print(f"\nPOOLED edge gap (fixed - current) at row positions 0-{EDGE - 1}: {gap:+.6f} "
          f"[footnote: the subset is not the corpus]")
    print(f"rest-of-row gap, same rows: {gap_rest:+.6f}")
    print(out["reading"])
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
