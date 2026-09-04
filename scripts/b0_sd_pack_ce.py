#!/usr/bin/env python3
"""What cross-document leakage costs in the unit b0-23 and Stage D actually report.

TWO QUESTIONS, and they are different. e1-32 measured the first; only the second decides whether
a per-block CE delta means what its fact says it means.

(1) THE INVARIANT (e1's, reproduced here independently on a different checkpoint and in my own
    code): with cu_seqlens passed, does a document's mixer output inside a packed row equal the
    same document scored alone? e1 read max 48.9 against a 0.93 tolerance at block 0, two-document
    control 35.75, decaying from each document's start, doc 0 clean. If that reproduces on
    ckpt_b0_sd_unlooped.pt then it is a property of the kernel or the plumbing, not of one run.

(2) THE ONE THAT CHANGES A FACT'S READING: eval/domain_loss.py:229 calls `model(x[i:i+bs])` with
    NO cu argument, and HybridLM.forward defaults cu=None. So the production eval path does not
    merely leak state across documents despite cu -- it never declares the boundaries at all,
    and its attention is full-row causal across every document in the 4096-token row. Every
    number in facts measured through domain_loss (ds.n2_params_vs_data_matched_compute, every
    eff.* per-domain loss, both Stage D legs) was taken on that path.

    So the question for the boundary line is not "is there leakage" -- there is, by construction,
    and more of it than e1's probe measures. It is HOW BIG the leakage is in CE per token, because
    that is the unit the deltas are in. Measured by scoring the same val rows twice on the same
    checkpoint, once as the eval path does (cu=None) and once with cu=doc_cu_seqlens(rows), and
    differencing per block. A delta of -0.0108 nat (N2's) sits against whatever this prints.

WHAT THIS DOES NOT DO: it does not decide whether cu SHOULD be passed in eval. Both legs of every
comparison went through the identical path, so no published delta moves; changing the path would
rescore everything against a different reference. That is a ruling for the fix owner (e1 names the
line, 6e rules), and this script deliberately stops at the measurement.

USAGE
    CUDA_VISIBLE_DEVICES=<the granted card> python3 scripts/b0_sd_pack_ce.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

CKPT = "ckpt_b0_sd_unlooped.pt"   # my own Stage D arm: independent of e1's ckpt_n7c_p3.pt
# A DOMAIN THE ARMS ACTUALLY TRAINED ON AND WHOSE CACHE IS ALREADY WARM. "code" is neither: it is
# not in data/mix_scale_3.24b.json's nine domains, and its tokens_code.pt.vocab is empty, so
# val_seqs refuses rather than retokenizing under a live run (cache_guard, correctly).
DOMAINS = ["math_owm_stage2", "en_c4_stage2", "cot", "textbook_30b", "chatml", "chat_qa",
           "zh_web", "code_py_starcoder", "code_py_rp1t"]
INVARIANT_DOMAIN = "code_py_starcoder"   # which domain's row 0 the packed-vs-alone control uses
N_ROWS = 32                       # 32 x 4096 rows is enough for a per-block mean; not a fact


def main():
    import torch  # noqa: PLC0415
    from tokenizers import Tokenizer  # noqa: PLC0415

    import model as M  # noqa: PLC0415
    from scripts.loader import load_checkpoint  # noqa: PLC0415
    from train import doc_cu_seqlens  # noqa: PLC0415

    if not M.HAS_FA:
        raise SystemExit("REFUSING: HAS_FA is False, so the varlen path cu would select is absent "
                         "and a cu=None-vs-cu comparison would compare one path to itself.")

    mdl, cfg = load_checkpoint(CKPT, dtype=torch.bfloat16)
    mdl = mdl.cuda().eval()
    tok = Tokenizer.from_file(os.path.join(ROOT, "data", "tokenizer.json"))
    eos = tok.token_to_id("<eos>")
    kinds = ["MLA" if isinstance(b.mixer, M.GatedMLA) else "KDA" for b in mdl.blocks]
    print(f"{CKPT}: {len(mdl.blocks)} blocks " + " ".join(f"{i}:{k}" for i, k in enumerate(kinds)))
    print(f"attn_res={getattr(cfg, 'attn_res', '?')} doc_mask={getattr(cfg, 'doc_mask', '?')} "
          f"seq={getattr(cfg, 'seq', '?')}")

    # ---------- (2) FIRST, because it is the one that decides the boundary line ----------
    sys.path.insert(0, os.path.join(ROOT, "eval"))
    from cache_guard import set_vocab_id  # noqa: PLC0415
    from domain_loss import val_seqs  # noqa: PLC0415

    # WITHOUT THIS the guard reads train.VOCAB_ID as None, compares it against a real stamp, and
    # refuses -- exactly the failure cache_guard.py's own docstring records. scripts.loader
    # .load_checkpoint does not set it; only train.build_mix and this call do.
    set_vocab_id(cfg)

    # ALL NINE DOMAINS, because the re-score ruling is about every published per-domain value and a
    # one-domain number cannot say whether the artifact is uniform across them. The per-domain
    # spread is the part that decides whether one correction factor exists or nine do.
    print("\n== the eval path (cu=None, domain_loss.py:229) against the same rows WITH cu declared")
    print(f"  {'domain':20s} {'rows':>5s} {'eos/row':>8s} {'cu=None':>9s} {'cu':>9s} "
          f"{'delta':>10s} {'SE':>9s} {'lower':>7s}")
    table, alld = [], []
    for dom in DOMAINS:
        rows = val_seqs(dom, tok)
        if rows is None:
            print(f"  {dom:20s} no val rows -- SKIPPED, not scored as zero")
            continue
        rows = rows[:N_ROWS]
        x, y = rows[:, :-1].cuda(), rows[:, 1:].cuda()
        med_eos = sorted((r == eos).sum().item() for r in rows)[len(rows) // 2]

        def ce_per_row(with_cu, x=x, y=y):
            out = []
            with torch.no_grad():
                for i in range(len(x)):
                    xi, yi = x[i:i + 1], y[i:i + 1]
                    cu = doc_cu_seqlens(xi, eos) if with_cu else None
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        lg = mdl(xi, cu=cu)
                    lg = lg[0] if isinstance(lg, tuple) else lg
                    ce = torch.nn.functional.cross_entropy(
                        lg.float().view(-1, lg.shape[-1]), yi.reshape(-1), reduction="sum")
                    out.append(float(ce) / yi.numel())
            return out

        # A RERUN BASELINE BEFORE ANY DIFF IS CALLED A DIFFERENCE, per domain rather than once:
        # a kernel that is deterministic on one shape is not thereby deterministic on another.
        a1, a2 = ce_per_row(False), ce_per_row(False)
        rerun = max(abs(p - q) for p, q in zip(a1, a2, strict=True))
        if rerun > 1e-4:
            raise SystemExit(f"REFUSING: {dom} forwarded twice differs by {rerun:.6f} nat/row, the "
                             f"size of the effect being measured. Nothing here is attributable.")
        b = ce_per_row(True)
        d = [q - p for p, q in zip(a1, b, strict=True)]
        n = len(d)
        mean = sum(d) / n
        sd = (sum((v - mean) ** 2 for v in d) / (n - 1)) ** 0.5 if n > 1 else 0.0
        se = sd / n ** 0.5 if n > 1 else 0.0
        lower = sum(1 for v in d if v < 0)
        table.append((dom, mean, se, sd, lower, n, sum(a1) / n, sum(b) / n, med_eos, rerun))
        alld += d
        print(f"  {dom:20s} {n:5d} {med_eos:8d} {sum(a1) / n:9.4f} {sum(b) / n:9.4f} "
              f"{mean:+10.6f} {se:9.6f} {lower:4d}/{n}")

    if not table:
        raise SystemExit("REFUSING: no domain produced rows, so there is no table.")
    # POOLED OVER ROWS, not a mean of the nine per-domain means: the rows are the sampling unit and
    # the domains have equal row counts here, so the two coincide -- stated because that coincidence
    # is an artifact of equal n, not a property of the statistic (same trap as row- vs
    # token-weighting agreeing on equal-length blocks).
    N = len(alld)
    M = sum(alld) / N
    SD = (sum((v - M) ** 2 for v in alld) / (N - 1)) ** 0.5
    SE = SD / N ** 0.5
    dmeans = [t[1] for t in table]
    print(f"\n  POOLED over {N} rows in {len(table)} domains: {M:+.6f} nat/token, sd {SD:.6f}, "
          f"SE {SE:.6f}, 95% [{M - 1.96 * SE:+.6f}, {M + 1.96 * SE:+.6f}], "
          f"{sum(1 for v in alld if v < 0)}/{N} rows lower with cu")
    print(f"  PER-DOMAIN SPREAD: min {min(dmeans):+.6f} ({table[dmeans.index(min(dmeans))][0]}), "
          f"max {max(dmeans):+.6f} ({table[dmeans.index(max(dmeans))][0]}), "
          f"range {max(dmeans) - min(dmeans):.6f} -- a range larger than the pooled mean means "
          f"there is no single correction factor and a re-score cannot be a constant shift")
    print(f"  AGAINST N2's OWN DELTA of -0.010770 nat: ratio {abs(M) / 0.010770:.2f}x. Both legs of "
          f"every published comparison took the cu=None path, so no delta moves; this is the size "
          f"of a common-mode artifact, and it is what the boundary line must state.")

    # ---------- (1) e1's invariant, reproduced independently ----------
    # Two documents only: the smallest packing that can violate it. Real corpus documents from the
    # same val rows, not HumanEval -- a second data source as well as a second checkpoint.
    print("\n== e1-32's invariant on this checkpoint: a document packed vs the same document alone")
    # RE-READ the domain rather than reusing whatever `rows` the loop above left bound: the loop
    # variable would silently make this section depend on which domain came last.
    inv = val_seqs(INVARIANT_DOMAIN, tok)
    if inv is None:
        raise SystemExit(f"REFUSING: {INVARIANT_DOMAIN} has no val rows for the control.")
    flat = inv[0].tolist()
    cuts = [i for i, t in enumerate(flat) if t == eos]
    if len(cuts) < 2:
        print("  row 0 holds fewer than 2 documents; skipping the control")
        return 0
    d0 = flat[:cuts[0]]
    d1 = flat[cuts[0] + 1:cuts[1]]
    if len(d0) < 8 or len(d1) < 8:
        print(f"  documents too short ({len(d0)}, {len(d1)}); skipping")
        return 0

    caught = {}

    def wrap(key, mod):
        inner = mod.forward

        def fwd(*a, **k):
            out = inner(*a, **k)
            caught[key] = (out[0] if isinstance(out, tuple) else out).detach().float()
            return out
        mod.forward = fwd
        return inner

    originals = [(b_.mixer, wrap(i, b_.mixer)) for i, b_ in enumerate(mdl.blocks)]

    def states(ids):
        # EVERY CAPTURE CHECKED, because a wrap that misses the taken path prints a clean
        # all-clear from an empty dict (e1 hit exactly that under attn_res).
        caught.clear()
        t = torch.tensor([ids], device="cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            mdl(t, cu=doc_cu_seqlens(t, eos))
        if len(caught) != len(mdl.blocks):
            raise SystemExit(f"REFUSING: captured {len(caught)} of {len(mdl.blocks)} mixers; any "
                             f"verdict from this probe would be vacuous.")
        return {i: v[0].clone() for i, v in caught.items()}

    packed = states([*d0, eos, *d1, eos])
    solo1 = states([*d1, eos])
    off = len(d0) + 1
    print(f"  doc0 {len(d0)} tok, doc1 {len(d1)} tok at row offset {off}")
    print(f"  {'blk':>3s} {'kind':4s} {'absmean':>9s} {'tol':>7s} {'max':>9s} {'first3':>8s} "
          f"{'rest':>8s} {'last10':>8s}  shape")
    first = None
    for i in sorted(packed):
        a = packed[i][off:off + len(d1)]
        s = solo1[i][:len(d1)]
        diff = (a - s).abs().amax(dim=-1)
        am = solo1[i].abs().mean().item()
        tol = max(0.05, 0.25 * am)   # e1's scale rule: a quarter of the layer's own mean
        mx, f3 = diff.max().item(), diff[:3].max().item()
        rest = diff[3:].max().item() if len(d1) > 3 else 0.0
        l10 = diff[-10:].max().item()
        shape = ("agrees" if mx <= tol else
                 "boundary only (<=3 pos, k=4 conv)" if rest <= tol else
                 "decaying from the start" if l10 < 0.25 * f3 else "uniform")
        print(f"  {i:3d} {kinds[i]:4s} {am:9.4f} {tol:7.3f} {mx:9.4f} {f3:8.3f} {rest:8.3f} "
              f"{l10:8.3f}  {shape}")
        if first is None and mx > tol:
            first = (i, kinds[i], mx, tol, shape)
    if first is None:
        print("  ALL BLOCKS AGREE -- the invariant HOLDS here, which CONTRADICTS e1-32 and makes "
              "the checkpoint or the data source the difference, not the kernel.")
    else:
        i, k, mx, tol, shape = first
        print(f"  FIRST DIVERGENCE: block {i} ({k}), max {mx:.4f} against tol {tol:.3f}, {shape}"
              f" -- REPRODUCES e1-32 on a second checkpoint and a second data source.")
    for mod, inner in originals:
        mod.forward = inner
    return 0


if __name__ == "__main__":
    sys.exit(main())
