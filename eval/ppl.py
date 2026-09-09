#!/usr/bin/env python3
"""Held-out perplexity, reported PER DOMAIN.

Rebuilds exactly the rows train.py holds out (same caches, same val_frac, same
val_rows_max) and scores each domain on its own; the summary is an unweighted
mean across domains, because a row-weighted blend is dominated by the largest
domain and cannot show a small one moving.

    python eval/ppl.py --ckpt ckpt_k5_clean_0827.pt --tokenizer data/tokenizer_k5.json
"""

import argparse
import math
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "eval"))

from scripts.loader import EOS_ID, load_checkpoint, load_tokenizer  # noqa: E402


def selftest():
    """KNOWN ANSWER for the holdout split, on hand-built mixes. No GPU, no caches.

    THE DEFECT THIS EXISTS FOR: until 2026-09-08 this file computed its own split from
    the global Cfg.val_frac while train.py honoured each domain's `val_frac` key, so on
    any mix with a per-domain override the two disagreed and ppl.py scored TRAINED rows
    while its docstring promised held-out ones. Five mixes carry such an override today
    (data/mix_e1_{control_arm,n1,n8,n64,n256}.json -- p_format in all five, plus each
    arm's own s_inject_n*), so this is not a hypothetical shape.

    EXACT COUNTS, NOT "SMALLER". The two arms of the bug differ by a handful of rows on
    a small domain, and an inequality assertion passes on both. Case 1 is the negative
    control for case 2: same rows, same global val_frac, one key added, and the answer
    must move 5 -> 0. Case 3 pins the max(1, ...) that a zero GLOBAL val_frac must still
    produce -- test_plan_length depends on it, and collapsing the two zeroes into one
    branch shifted every pool by a row.
    """
    import train

    fails = []

    def check(name, got, want):
        ok = got == want
        print(f"  {'ok  ' if ok else 'FAIL'} {name}: got {got}, want {want}")
        if not ok:
            fails.append(name)

    train.Cfg.val_frac = 0.05
    train.Cfg.val_rows_max = 5000
    plain = {"domains": {"d": {"weight": 1.0}}}
    zero = {"domains": {"d": {"weight": 1.0, "val_frac": 0}}}
    half = {"domains": {"d": {"weight": 1.0, "val_frac": 0.5}}}

    check("no key -> global 5% of 100", train.val_split_n("d", 100, plain), 5)
    check("val_frac 0 -> exactly 0 rows held out", train.val_split_n("d", 100, zero), 0)
    check("per-domain 0.5 overrides the global", train.val_split_n("d", 100, half), 50)
    check("the cap applies to a per-domain frac too",
          train.val_split_n("d", 10 ** 7, half), 5000)

    train.Cfg.val_frac = 0.0
    check("global 0.0 still holds back one row (test_plan_length depends on it)",
          train.val_split_n("d", 100, plain), 1)
    check("an explicit 0 KEY is not the same as a global 0.0",
          train.val_split_n("d", 100, zero), 0)

    # The real mixes, so the case dies if a mix stops carrying the override.
    import glob
    import json as _json
    train.Cfg.val_frac = 0.05
    n_zero = 0
    for p in sorted(glob.glob(os.path.join(ROOT, "data", "mix_e1_*.json"))):
        mix = _json.load(open(p, encoding="utf-8"))
        for name, dcfg in mix["domains"].items():
            if isinstance(dcfg, dict) and dcfg.get("val_frac") == 0:
                n_zero += 1
                got = train.val_split_n(name, 1000, mix)
                if got != 0:
                    fails.append(f"{os.path.basename(p)}:{name}")
                    print(f"  FAIL {os.path.basename(p)}:{name} held out {got} rows")
    check("the shipped e1 mixes still carry val_frac 0 domains", n_zero >= 5, True)
    print(f"  ({n_zero} val_frac:0 domain(s) across data/mix_e1_*.json, all held out 0 rows)")

    print(f"\n{'ALL OK' if not fails else 'FAILED: ' + ', '.join(map(str, fails))}")
    return len(fails)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--tokenizer", default=os.path.join(ROOT, "data", "tokenizer.json"))
    ap.add_argument("--mix", default=os.path.join(ROOT, "data", "mix_scale_3.24b.json"))
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--rows", type=int, default=512, help="val rows scored per domain")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--cu_path", choices=["cu_none", "doc_cu"], default="cu_none",
                    help="doc_cu passes the document mask; cu_none is what every published "
                         "ppl figure was taken with (audit_0904 E10). train._domain_seqs "
                         "returns PACKED rows, so the two differ.")
    a = ap.parse_args()

    if a.selftest:
        sys.exit(1 if selftest() else 0)
    if not a.ckpt:
        ap.error("--ckpt required (unless --selftest)")

    import json

    import train
    from train import doc_cu_seqlens
    from cache_guard import guard

    model, cfg = load_checkpoint(a.ckpt, device=a.device)
    tok = load_tokenizer(a.tokenizer, cfg)
    for p in model.parameters():
        p.data = p.data.contiguous()
    # _domain_seqs reads Cfg for seq/fone/num_id, so the checkpoint's own config drives it.
    for k, v in vars(cfg).items():
        if not k.startswith("_"):
            setattr(train.Cfg, k, v)

    mix = json.load(open(a.mix, encoding="utf-8"))
    # Before the first _domain_seqs call, not inside the loop: this ran on card 7 against
    # the live 20B run's nine caches and printed "cache was built by another vocabulary,
    # retokenizing" two minutes in (fb killed it by exact PID, 2026-09-02). train.VOCAB_ID
    # is set only by train.build_tokenizer; load_checkpoint never touches it, so it was
    # None, every stamp read as a mismatch, and the rebuild would have re-stamped nine
    # training caches with an empty vocabulary. Cfg is set above first because the guard's
    # cache path depends on Cfg.fone.
    guard(cfg, list(mix["domains"]))
    out = {}
    for name in mix["domains"]:
        seqs = train._domain_seqs(name, tok, True, False)
        seqs = seqs[0] if train.Cfg.fone else seqs
        # CALL train's split, do not restate it. This line read the GLOBAL Cfg.val_frac
        # while train.py honours each domain's own `val_frac` key, so for the five
        # mix_e1_* arms -- every one sets p_format's val_frac to 0, plus its own
        # s_inject_n* domain -- this scored rows the run TRAINED on and printed them
        # under a docstring promising "exactly the rows train.py holds out". A domain
        # with val_frac 0 now yields 0 rows and is skipped by the `if not len(rows)`
        # below, which is what the run itself did. Found by 4c, 2026-09-08.
        n_val = train.val_split_n(name, len(seqs), mix)
        rows = seqs[:n_val][: a.rows].long()
        if not len(rows):
            continue
        X, Y = rows[:, :-1], rows[:, 1:]
        tot = ntok = 0.0
        with torch.no_grad():
            for i in range(0, len(X), a.batch):
                xb, yb = X[i : i + a.batch].to(a.device), Y[i : i + a.batch].to(a.device)
                # cu REACHES THE FORWARD. train._domain_seqs packs multiple documents per row,
                # so without the mask attention reads across the boundaries inside a row while
                # training used doc_cu_seqlens (E10: this file's docstring claimed it "rebuilds
                # exactly the rows train.py held out" while scoring them on a different path).
                cu = doc_cu_seqlens(xb, EOS_ID) if a.cu_path == "doc_cu" else None
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits, _ = model(xb, cu=cu) if cu is not None else model(xb)
                    loss = torch.nn.functional.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]).float(), yb.reshape(-1)
                    )
                n = yb.numel()
                tot += loss.item() * n
                ntok += n
        out[name] = tot / ntok
        print(
            f"  {name:<6} loss {out[name]:.4f}  ppl {math.exp(out[name]):7.2f}  ({len(X)} rows)", flush=True
        )

    if out:
        # Unweighted mean across domains, not the row-weighted blend train.py prints:
        # a blend hides a small domain moving.
        m = sum(out.values()) / len(out)
        print(f"  {'MEAN':<6} loss {m:.4f}  ppl {math.exp(m):7.2f}  (unweighted across domains)")


if __name__ == "__main__":
    main()
