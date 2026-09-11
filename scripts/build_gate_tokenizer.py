#!/usr/bin/env python3
"""Build the V4.1 gate-mix tokenizer candidates and choose 20K vs 32K (ae-5, fb
REBUILD ruling 2026-09-10).

Unfreeze condition 2 (material distribution change) was met on UltraData
(facts/tokenizer.json#tok.ultra_freeze_tax_0910, +8.5% tokens/byte), and the
rebuild cost is zero: no V4.1 checkpoint exists. This script fits TWO sizes on
ONE stratified EQUAL-BYTE sample over the domains the model actually reads --
UltraData L2/L3 plus the older code, math, CoT and English -- and scores each on
a DOCUMENT-DISJOINT held-out slice per domain. The vocabulary must be good at
all seven; an UltraData-only fit would under-serve math/CoT/English.

The sample balance is equal-byte, not the corpus mix (same reasoning as
build_tokenizer.py: it decides what earns merges). Fit and eval are a contiguous
read split by documents, so no eval doc appears in the fit (reshuffling one pool
gives identical order-invariant token counts -- the #225 review finding).

Outputs candidates only; it never writes data/tokenizer.json (ids do not survive
a rebuild). The chosen size is promoted by a separate step that re-pins the
specials and Cfg.

    python3 scripts/build_gate_tokenizer.py \
        --domains code_ultra_l2_sample,code_ultra_l3_sample,code_py_starcoder,\
code_py_rp1t,math_owm_stage2,cot,en_c4_stage2 \
        --sizes 20000,32768 --json runs/gate_tokenizer_choice.json
"""
# restartable: two deterministic BPE fits plus reads; an interrupt just re-runs
# (no accumulated state), so there is no per-shard write to make.
import argparse
import glob
import json
import os
import random
import sys

ROOT = os.environ.get("AUPAI_ROOT",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import tokenizer_report as R  # noqa: E402
from build_p1_tokenizer import fit_vocab  # noqa: E402
from build_tokenizer import CHAT_SPECIALS  # noqa: E402
from tokenizer_eval import gates  # noqa: E402

D_MODEL = 1024
SEED = 7


def read_domain(corpus, d, cap_bytes, rng):
    """content strings from corpus/<d>/*.jsonl, up to cap_bytes, shuffled within
    sampled shards."""
    fs = sorted(glob.glob(os.path.join(corpus, d, "*.jsonl")))
    if not fs:
        sys.exit(f"no shards under {os.path.join(corpus, d)}")
    fs = rng.sample(fs, min(len(fs), 12))
    rows, got = [], 0
    for f in fs:
        with open(f, encoding="utf-8") as fh:
            lines = fh.readlines()
        rng.shuffle(lines)
        for line in lines:
            try:
                c = json.loads(line).get("content", "")
            except json.JSONDecodeError:
                continue
            if c:
                rows.append(c)
                got += len(c.encode("utf-8"))
                if got >= cap_bytes:
                    return rows
    return rows


def split_at_bytes(rows, n):
    """first n bytes -> cut index (document boundary)."""
    used = 0
    for i, r in enumerate(rows):
        used += len(r.encode("utf-8"))
        if used >= n:
            return i + 1
    return len(rows)


def score(tok, rows):
    """tokens/byte and chars/token on held-out rows."""
    nb = sum(len(r.encode("utf-8")) for r in rows)
    nc = sum(len(r) for r in rows)
    nt = sum(len(tok.encode(r).ids) for r in rows)
    return {"tokens_per_byte": nt / nb, "chars_per_token": nc / nt, "n_docs": len(rows)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument(
        "--domains",
        default="code_ultra_l2_sample,code_ultra_l3_sample,code_py_starcoder,"
        "code_py_rp1t,math_owm_stage2,cot,en_c4_stage2",
    )
    ap.add_argument("--sizes", default="20000,32768")
    ap.add_argument("--fit_bytes", type=int, default=40_000_000, help="per domain")
    ap.add_argument("--eval_bytes", type=int, default=20_000_000, help="per domain, disjoint")
    ap.add_argument("--outdir", default="")
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    corpus = os.path.join(a.root, "data", "corpus")
    outdir = a.outdir or os.path.join(a.root, "data")
    domains = [d for d in a.domains.split(",") if d]
    sizes = [int(s) for s in a.sizes.split(",")]

    fit, ev, counts = {}, {}, {}
    for d in domains:
        rows = read_domain(corpus, d, a.fit_bytes + a.eval_bytes, random.Random(SEED))
        random.Random(SEED).shuffle(rows)
        cut = split_at_bytes(rows, a.fit_bytes)
        fit[d], ev[d] = rows[:cut], rows[cut:]
        counts[d] = {
            "n_docs": len(rows), "fit_docs": cut, "eval_docs": len(rows) - cut,
            "fit_bytes": sum(len(r.encode()) for r in rows[:cut]),
            "eval_bytes": sum(len(r.encode()) for r in rows[cut:]),
        }
    fit_texts = [r for d in domains for r in fit[d]]
    random.Random(SEED).shuffle(fit_texts)

    result = {
        "config": {"domains": domains, "sizes": sizes, "seed": SEED,
                   "fit_bytes_per_domain": a.fit_bytes, "eval_bytes_per_domain": a.eval_bytes,
                   "d_model": D_MODEL, "chat_specials": CHAT_SPECIALS, "counts": counts},
        "candidates": {},
    }

    for v in sizes:
        print(f"fitting V={v} on {len(fit_texts)} docs...", flush=True)
        tok = fit_vocab(v, fit_texts)
        path = os.path.join(outdir, f"tok_gate_v{v}.json")
        tok.save(path)
        entry = {"path": os.path.relpath(path, a.root), "vocab": tok.get_vocab_size(),
                 "per_domain": {}, "gates": gates(tok, {d: ev[d][:2000] for d in domains})}
        tb = nt_all = 0
        for d in domains:
            m = score(tok, ev[d])
            entry["per_domain"][d] = m
            tb += sum(len(r.encode()) for r in ev[d])
            nt_all += round(m["tokens_per_byte"] * sum(len(r.encode()) for r in ev[d]))
        entry["pooled_tokens_per_byte"] = nt_all / tb
        # embedding/head parameter cost at d=1024; Cfg.vocab pads vocab_real up to a
        # multiple of 64 for the aligned head kernel (fb: keep Cfg.vocab a multiple of 64)
        padded = ((v + 63) // 64) * 64
        entry["padded_cfg_vocab"] = padded
        entry["embed_params_tied_M"] = D_MODEL * padded / 1e6
        entry["embed_params_untied_M"] = 2 * D_MODEL * padded / 1e6
        entry["ref_fertility"] = R.ref_fertility(tok).get("ref fertility")
        result["candidates"][str(v)] = entry
        print(f"  pooled {entry['pooled_tokens_per_byte']:.5f} tok/byte, "
              f"tied {entry['embed_params_tied_M']:.2f}M params", flush=True)

    # the decision table
    print(f"\n{'domain':<24}" + "".join(f"{v:>14}" for v in sizes) + "   (tokens/byte held-out)")
    for d in domains:
        vals = [result["candidates"][str(v)]["per_domain"][d]["tokens_per_byte"] for v in sizes]
        print(f"{d:<24}" + "".join(f"{x:>14.5f}" for x in vals))
    print(f"{'POOLED':<24}" + "".join(
        f"{result['candidates'][str(v)]['pooled_tokens_per_byte']:>14.5f}" for v in sizes))
    for v in sizes:
        e = result["candidates"][str(v)]
        print(f"V={v}: padded {e['padded_cfg_vocab']}, tied {e['embed_params_tied_M']:.2f}M, "
              f"untied {e['embed_params_untied_M']:.2f}M, ref fert {e['ref_fertility']}, "
              f"gates {json.dumps(e['gates'])}")

    if a.json:
        with open(a.json, "w") as fh:
            json.dump(result, fh, indent=1)
        print(f"wrote {a.json}")


if __name__ == "__main__":
    main()
