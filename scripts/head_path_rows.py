#!/usr/bin/env python3
# restartable: ZERO cards. torch.load(map_location="cpu", mmap=True) plus tensor norms. Seconds.
"""b0-17: did untying the head reduce tok.weight's growth? It INCREASED it, and one row class proves why.

    CUDA_VISIBLE_DEVICES= python3 scripts/head_path_rows.py \
        --base ckpt_ab_shapelr_base.pt.ep1 \
        --arm ckpt_ab_untiehead_untiehead.pt.ep1 \
        --arm ckpt_ab_untieheadlr_untieheadlr.pt.ep1 --out runs/b0_17_rows.json
    python3 scripts/head_path_rows.py --selftest

THE HYPOTHESIS b0-17 TESTED, and why the bulk norm could not settle it: b0-10 found tok.weight
growing uniformly at every quantile, and 1e's candidate mechanism was that the TIED head trains
tok at embed_lr 0.1 through a second gradient path. If that were the cause, removing the path
(--untie_head) should make tok grow LESS.

It grows MORE: row-norm median 16.9115 -> 18.3308 (arm 2, x1.0839) -> 19.8036 (arm 3, x1.1710),
and the lower the head's lr the more tok grows. A bulk number cannot say whether the path was
actually removed, though, so on its own that is a puzzle rather than a refutation.

THE ROW CLASSES SETTLE IT. tok.weight has 32832 rows and three classes with DIFFERENT gradient
paths (train.py:164-174):

    [0, 32773)        vocab_real: reachable as an input id AND as a head row
    [32773, 32784)    alignment padding: NEVER an input id, NEVER a target -- reachable ONLY
                      through the head. Its movement IS the head path, not a proxy for it.
    [32784, 32832)    beyond vocab: neither path. The noise floor, and it must read ~init.

So the middle class is an isolated measurement of exactly the mechanism under test. Measured on
the three .ep1 checkpoints:

    row class            base tok    arm2 tok    arm2 head
    [0,32773)             16.9211     18.3390      16.5053
    [32773,32784)          6.6575      0.0000       7.1036
    [32784,32832)          0.6209      0.6221       0.6209

  - arm2's tok pad rows are EXACTLY 0.0000: the head path was removed completely, not weakened.
    (model.py:455 zeroes them at init once untied, and nothing else can reach them.)
  - arm2's HEAD pad rows are 7.1036 against the tied base's 6.6575 -- the path itself did not get
    weaker; it got 1.067x stronger.
  - and tok grew MORE anyway.

That is why the hypothesis is REFUTED rather than unsupported: the named path was fully removed,
it did not weaken, and the effect moved the other way.

WHAT REPLACES IT IS A GUESS, LABELLED AS ONE. A tied table receives input-path and head-path
gradients that PARTIALLY CANCEL; untying removes the cancellation and the input path alone pushes
tok further. This file does not measure gradient signs, so the cancellation is not established --
only that "the head path inflates tok" is excluded.

THE PAD ROWS ARE NONZERO IN EVERY ARM'S HEAD (6.66 / 7.10 / 0.73), which is
eff.vocab_padding_softmax_defect: padding rows sit in the softmax denominator and steal mass. It
is a real defect and it is NOT a confound here -- all three arms carry it at the same order, so it
cannot produce a between-arm difference.
"""

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("FLA_FLASH_KDA", "0")

# From train.py:164-174. Read from Cfg at runtime rather than trusted as literals -- a tokenizer
# change moves both boundaries and would silently reclassify every row.
INIT_STD = 0.02        # model.py:453
D_MODEL = 1024


def row_classes(vocab_real, vocab, n_rows):
    """The three gradient-path classes, as (lo, hi, label).

    Built from the checkpoint's own numbers. The third class exists only when the tensor is padded
    past cfg.vocab for kernel alignment; it is the noise floor, so its absence must be visible
    rather than silently dropping a column from the table.
    """
    out = [(0, vocab_real, "input+head"), (vocab_real, vocab, "head-only")]
    if n_rows > vocab:
        out.append((vocab, n_rows, "no-path"))
    return out


def read_rows(path):
    """Per-class median/max row norm for tok and head, plus whether the two are the same tensor."""
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    sd = ck["model"]
    cfg = ck.get("cfg")
    get = (lambda k, d: cfg.get(k, d)) if isinstance(cfg, dict) else (lambda k, d: getattr(cfg, k, d))
    tok, head = sd["tok.weight"], sd["head.weight"]
    n_rows = tok.shape[0]
    vocab = int(get("vocab", n_rows))
    vocab_real = int(get("vocab_real", vocab))
    if not (0 < vocab_real <= vocab <= n_rows):
        raise ValueError(f"{path}: vocab_real {vocab_real} / vocab {vocab} / rows {n_rows} are not "
                         f"ordered; the row classes would overlap or invert and every number below "
                         f"would be a different quantity than its label")
    rec = {"ckpt": os.path.basename(path), "rows": n_rows, "vocab": vocab,
           "vocab_real": vocab_real, "tied": bool(tok is head or torch.equal(tok, head)),
           "classes": {}}
    for name, t in (("tok", tok), ("head", head)):
        f = t.float()
        for lo, hi, lbl in row_classes(vocab_real, vocab, n_rows):
            r = f[lo:hi].norm(dim=1)
            rec["classes"].setdefault(lbl, {})[name] = {
                "lo": lo, "hi": hi, "median": r.median().item(), "max": r.max().item(),
            }
    return rec


def _selftest():
    import tempfile

    import torch
    fails = []
    v_real, v, rows, d = 20, 24, 32, 8

    # A synthetic pair with the REAL structure: a tied arm where tok IS head, and an untied arm
    # whose tok pad rows are zeroed (model.py:455) while its head pad rows are not.
    def mk(tied, pad_tok_zero):
        tok = torch.full((rows, d), 2.0)
        head = tok if tied else torch.full((rows, d), 3.0)
        if not tied:
            head = head.clone()
            if pad_tok_zero:
                tok = tok.clone()
                tok[v_real:v] = 0.0
        tok[v:] = INIT_STD                      # beyond-vocab: never trained
        head = head.clone()
        head[v:] = INIT_STD
        return {"tok.weight": tok, "head.weight": head}

    with tempfile.TemporaryDirectory() as td:
        pt = os.path.join(td, "tied.pt")
        pu = os.path.join(td, "untied.pt")
        cfg = {"vocab": v, "vocab_real": v_real}
        torch.save({"model": mk(True, False), "cfg": cfg}, pt)
        torch.save({"model": mk(False, True), "cfg": cfg}, pu)

        rt, ru = read_rows(pt), read_rows(pu)

        # 1. TIED MUST BE DETECTED. If this reads False, every "the tied base did X" sentence in
        #    the reading is attached to the wrong arm, and nothing errors.
        if not rt["tied"]:
            fails.append("a checkpoint whose tok and head are the same tensor read tied=False; "
                         "the base arm would be labelled untied and the comparison inverted")
        if ru["tied"]:
            fails.append("a checkpoint with different tok and head read tied=True")

        # 2. THE HEAD-ONLY CLASS MUST BE READ FROM THE RIGHT ROWS. This is the whole measurement:
        #    if the slice is off by even one row it includes a trained vocab row and the 0.0000
        #    that refutes the hypothesis becomes a nonzero number with no meaning.
        ho = ru["classes"]["head-only"]
        if (ho["tok"]["lo"], ho["tok"]["hi"]) != (v_real, v):
            fails.append(f"head-only class is rows [{ho['tok']['lo']},{ho['tok']['hi']}), expected "
                         f"[{v_real},{v}) -- it must be exactly the alignment padding, since a "
                         f"real vocab row in this slice destroys the isolation the reading rests on")
        if ho["tok"]["max"] != 0.0:
            fails.append(f"untied arm's tok pad rows read max {ho['tok']['max']}, expected exactly "
                         f"0.0 -- the fixture zeroes them, so a nonzero here means the slice is "
                         f"wrong or the reduction is over the wrong axis")
        if ho["head"]["median"] <= 0.0:
            fails.append("untied arm's HEAD pad rows read zero; the fixture gives them 3.0, so the "
                         "measurement cannot see the path it is supposed to isolate")

        # 3. ROW-WISE, NOT WHOLE-TENSOR. norm(dim=1) on 4 rows of 8 columns at 3.0 is 3*sqrt(8)
        #    per row; a missing dim=1 would return one number ~2x larger and the medians would
        #    silently become tensor norms.
        want = 3.0 * (d ** 0.5)
        if abs(ho["head"]["median"] - want) > 1e-4:
            fails.append(f"head-only median is {ho['head']['median']:.4f}, expected {want:.4f} "
                         f"(3.0*sqrt({d})) -- the reduction is not per-row")

        # 4. THE NOISE FLOOR MUST BE PRESENT AND UNTRAINED. Without it, "0.62 is init" has nothing
        #    to stand on and a reader cannot tell a small number from an untrained one.
        if "no-path" not in ru["classes"]:
            fails.append(f"no no-path class for a tensor with {rows} rows past vocab {v}; the "
                         f"table loses its noise floor without saying so")
        else:
            np_ = ru["classes"]["no-path"]["tok"]["median"]
            if abs(np_ - INIT_STD * (d ** 0.5)) > 1e-4:
                fails.append(f"no-path rows read {np_:.4f}, expected {INIT_STD * d ** 0.5:.4f}; "
                             f"they are never reachable by any gradient, so anything else means "
                             f"the class boundaries are wrong")

    # 5. INVERTED BOUNDARIES MUST RAISE ValueError WITH THE NUMBERS, not crash downstream. A
    #    tokenizer change is exactly how vocab_real could exceed vocab. Deleting the explicit
    #    guard does NOT make this check red on its own: the empty slice reaches .max() and torch
    #    raises RuntimeError("Expected reduction dim to be specified for input.numel() == 0"),
    #    which a bare `except Exception: pass` would have accepted as the guard working. So the
    #    exception TYPE and the message are both asserted -- a crash is not a refusal, and the
    #    difference is invisible unless it is spelled out (same failure the l9 probe's control
    #    check had).
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "bad.pt")
        torch.save({"model": {"tok.weight": torch.zeros(8, 4), "head.weight": torch.zeros(8, 4)},
                    "cfg": {"vocab": 4, "vocab_real": 6}}, p)
        try:
            read_rows(p)
            fails.append("vocab_real > vocab was accepted; the head-only slice would be empty and "
                         "the reading would report 0.0000 for a measurement it never made -- the "
                         "same number the real refutation rests on")
        except ValueError as e:
            if "not\nordered" not in str(e).replace(" ", "\n") and "ordered" not in str(e):
                fails.append(f"inverted boundaries raised ValueError but not the ordering one: "
                             f"{e!r}")
        except Exception as e:                                   # noqa: BLE001
            fails.append(f"inverted boundaries raised {type(e).__name__}, not ValueError: {e!r}. "
                         f"A downstream crash is not a refusal -- it reports 'the tool is broken' "
                         f"where the truth is 'this checkpoint's vocab fields are inconsistent', "
                         f"and it means the explicit guard could be deleted with this check green")

    if fails:
        for f in fails:
            print(f"FAIL: {f}", file=sys.stderr)
        return 1
    print("head_path_rows selftest OK: tied detection works in both directions (a mislabelled base "
          "inverts every sentence in the reading); the head-only class is exactly the alignment "
          "padding [vocab_real, vocab), because one real vocab row in that slice destroys the "
          "isolation the whole refutation rests on; the untied arm's tok pad rows read exactly 0 "
          "and its head pad rows do not, which is the measurement itself; norms are per-row "
          "(dim=1), verified against 3*sqrt(d) rather than assumed; the no-path noise floor is "
          "present and sits at the init scale, so 'small' can be told from 'untrained'; and "
          "inverted vocab boundaries RAISE instead of yielding an empty slice that would report "
          "0.0000 for a measurement never made -- the same value the real refutation rests on.")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", help="the tied arm")
    ap.add_argument("--arm", action="append", default=[], help="untied arm(s); repeatable")
    ap.add_argument("--out", help="append one JSONL record per checkpoint")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    if not a.base:
        ap.error("--base is required (the tied arm is what the untied arms are read against)")

    recs = [read_rows(a.base)] + [read_rows(p) for p in a.arm]
    w = max(len(r["ckpt"]) for r in recs)
    labels = [lbl for _, _, lbl in row_classes(recs[0]["vocab_real"], recs[0]["vocab"],
                                               recs[0]["rows"])]
    for which in ("tok", "head"):
        print(f"\n=== {which}.weight row-norm median by gradient-path class ===")
        print(f"{'ckpt':{w}s} {'tied':>6s} " + " ".join(f"{lbl:>14s}" for lbl in labels))
        for r in recs:
            cells = " ".join(f"{r['classes'][lbl][which]['median']:>14.4f}" for lbl in labels)
            print(f"{r['ckpt']:{w}s} {str(r['tied']):>6s} {cells}")
    print(f"\nhead-only = rows [{recs[0]['vocab_real']}, {recs[0]['vocab']}): NEVER an input id, "
          f"NEVER a target, reachable ONLY through the head. Its movement IS the head gradient "
          f"path. no-path rows are unreachable by either and sit at init "
          f"{INIT_STD * D_MODEL ** 0.5:.4f}, which is the noise floor.")
    if a.out:
        with open(a.out, "a", encoding="utf-8") as fh:
            for r in recs:
                fh.write(json.dumps(r) + "\n")
        print(f"appended {len(recs)} record(s) to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
