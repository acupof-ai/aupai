"""Storage-granularity view of checkpoint-to-checkpoint parameter updates.

WHAT IT MEASURES. For two adjacent checkpoints of the same run, over the `model` state dict:
  moved    -- elements whose STORED value differs. bf16 in-place storage is the only record of
              an update, so this is exactly the quantity a bf16-absorbing update would drive to
              zero. Detection is exact inequality on the stored value.
  rel_l2   -- ||W_b - W_a||_2 / ||W_a||_2, computed in float32 on the upcast values.
  deciles  -- the same moved fraction bucketed by |w| at the EARLIER checkpoint, per parameter
              group. This is the discriminator between two explanations of a low `moved`:
                * a signed random walk that accumulates across steps -> a FLAT decile profile;
                * in-place round-to-nearest with NO residual, where each step faces the element's
                  own RELATIVE half-ulp and a step below it is discarded entirely -> a profile
                  that COLLAPSES as |w| grows, because half-ulp grows with |w|.
              Measured on v41_ced_0923 at the tail lr the profile collapses (60.6% in decile 1 to
              0.000% in deciles 8-10); at the mid lr it is nearly flat (70.4% to 63.0%). A
              random-walk explanation predicted flat in both.

THE CONTROL IS MANDATORY and runs first: the same file against itself must give moved 0 and
rel_l2 0.0 in every group. A comparison that cannot return 0 on identical inputs is measuring
itself, and returns non-zero here rather than producing a table.

mmap + CPU + read-only: nothing is written, and the checkpoints are never modified. Both sides
are held at once (~26 GB for the 13 GB pair), so run one pair at a time, pinned.

Usage:
    python3 scripts/ckpt_update_granularity.py --a CKPT_A --b CKPT_B [--group PATTERN] [--deciles]
    python3 scripts/ckpt_update_granularity.py --control CKPT [--group PATTERN] [--deciles]
"""
# restartable: read-only single pass over two already-materialized checkpoints; no cumulative
# state, so an interrupt loses only the groups not yet accumulated. Nothing is written.
import argparse
import collections
import gc
import re
import sys

import torch

# name -> regex over the state-dict key, grouped by what the parameters do
GROUPS = [
    ("moe_experts_w13", r"blocks\.\d+\.ffn\.w13$"),
    ("moe_experts_w2", r"blocks\.\d+\.ffn\.w2$"),
    ("moe_router", r"blocks\.\d+\.ffn\.router\.weight$"),
    ("moe_shared_ffn", r"blocks\.\d+\.ffn\.sh(13|2)\.weight$"),
    ("moe_expert_bias", r"blocks\.\d+\.ffn\.expert_bias$"),
    ("attn_qg", r"blocks\.\d+\.mixer\.qg\.weight$"),
    ("attn_kv", r"blocks\.\d+\.mixer\.(kv_down|kv_up)\.weight$"),
    ("attn_o", r"blocks\.\d+\.mixer\.o\.weight$"),
    ("csa_compress", r"blocks\.\d+\.mixer\.csa\.(compress_k|compress_v)\.(weight|bias)$"),
    ("csa_indexer", r"blocks\.\d+\.mixer\.csa\.(ik_weight|indexer_q\.weight)$"),
    ("csa_wkv", r"blocks\.\d+\.mixer\.csa\.w_(kv|z)\.weight$"),
    ("norm_gains", r"(blocks\.\d+\.n[12]\.g|^norm\.g)$"),
    ("embed_head", r"(^tok\.weight|^head\.weight)$"),
]


def group_of(key):
    for name, pat in GROUPS:
        if re.search(pat, key):
            return name
    return "OTHER"


def _load(path):
    return torch.load(path, map_location="cpu", mmap=True, weights_only=False)["model"]


def _decile_profile(ka, kb, nb=10):
    """moved fraction per |w| decile of the earlier tensor, plus each decile's median |w| and ulp."""
    a = ka.view(-1)
    b = kb.view(-1)
    aw = a.to(torch.float32).abs()
    with torch.no_grad():
        nz = aw > 0
        ex = torch.zeros_like(aw)
        ex[nz] = torch.floor(torch.log2(aw[nz]))
        ulp = torch.pow(2.0, ex - 7.0)  # bf16: 8 explicit mantissa bits
    order = torch.argsort(aw)
    moved = a != b
    n = aw.numel()
    out = []
    for i in range(nb):
        lo, hi = i * n // nb, (i + 1) * n // nb
        idx = order[lo:hi]
        if idx.numel() == 0:
            continue
        out.append((int(moved[idx].sum()), int(idx.numel()),
                    float(aw[idx].median()), float(ulp[idx].median())))
    return out


def compare(a_path, b_path, only=None, deciles=False, nb=10):
    A, B = _load(a_path), _load(b_path)
    if set(A) != set(B):
        sys.exit("checkpoints have different key sets -- not the same model")
    acc = collections.defaultdict(lambda: [0, 0, 0.0, 0.0])
    dec = collections.defaultdict(lambda: [[0, 0, [], []] for _ in range(nb)])
    ntensors = 0
    for k in A:
        va = A[k]
        if not hasattr(va, "dtype") or va.dtype not in (torch.bfloat16, torch.float32):
            continue
        if only and not re.search(only, k):
            continue
        ntensors += 1
        vb = B[k]
        g = group_of(k)
        moved = int((va.view(-1) != vb.view(-1)).sum())
        fa, fb = va.to(torch.float32).view(-1), vb.to(torch.float32).view(-1)
        d = fb - fa
        e = acc[g]
        e[0] += moved
        e[1] += va.numel()
        e[2] += float(torch.dot(d, d))
        e[3] += float(torch.dot(fa, fa))
        if deciles:
            for i, (m, t, wmed, umed) in enumerate(_decile_profile(va, vb, nb)):
                dec[g][i][0] += m
                dec[g][i][1] += t
                dec[g][i][2].append(wmed)
                dec[g][i][3].append(umed)
    rows = []
    for g, _ in GROUPS:
        if g not in acc:
            continue
        m, n, dn2, wn2 = acc[g]
        rel = (dn2 ** 0.5) / (wn2 ** 0.5) if wn2 > 0 else float("nan")
        rows.append((g, n, m, 100.0 * m / n, rel))
    tm = sum(r[2] for r in rows)
    tn = sum(r[1] for r in rows)
    return rows, tm, tn, dec


def print_table(rows, tm, tn, label):
    print(f"\n=== {label}  ({len(rows)} groups)")
    print(f"{'group':18}{'elems':>14}{'moved':>14}{'moved%':>10}{'rel_l2':>12}")
    for g, n, m, pct, rel in rows:
        print(f"{g:18}{n:>14}{m:>14}{pct:>9.3f}%{rel:>12.3e}")
    print(f"{'TOTAL':18}{tn:>14}{tm:>14}{100.0 * tm / tn:>9.3f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a")
    ap.add_argument("--b")
    ap.add_argument("--control", help="one checkpoint, compared with ITSELF; must be all-zero")
    ap.add_argument("--group", help="regex; restrict to matching parameter names")
    ap.add_argument("--deciles", action="store_true")
    ap.add_argument("--nb", type=int, default=10)
    ap.add_argument("--expect-zero", action="store_true",
                    help="exit non-zero if anything moved (what --control ought to do)")
    args = ap.parse_args()

    if args.control:
        rows, tm, tn, dec = compare(args.control, args.control, args.group, args.deciles, args.nb)
        print_table(rows, tm, tn, f"CONTROL: {args.control} vs ITSELF")
        if args.deciles:
            for g, prof in dec.items():
                if not args.group or g != "OTHER":
                    continue
        if tm != 0 or any(r[4] != 0.0 for r in rows):
            sys.exit("CONTROL FAILED: identical inputs produced non-zero differences -- the "
                     "comparison is measuring itself, so no table from it is usable")
        print("control OK: 0 moved, rel_l2 0.0 everywhere")
        return

    if not (args.a and args.b):
        ap.error("--a and --b, or --control")
    rows, tm, tn, dec = compare(args.a, args.b, args.group, args.deciles, args.nb)
    print_table(rows, tm, tn, f"{args.a}  ->  {args.b}")
    if args.deciles:
        for g in sorted(dec):
            prof = dec[g]
            if not any(p[1] for p in prof):
                continue
            print(f"\n  --- {g}: moved% by |w| decile")
            print(f"  {'dec':>4}{'elems':>14}{'moved':>14}{'moved%':>10}{'med|w|':>12}{'med ulp':>12}")
            for i, (m, t, ws, us) in enumerate(prof):
                if t == 0:
                    continue
                ws, us = sorted(ws), sorted(us)
                print(f"  {i + 1:>4}{t:>14}{m:>14}{100.0 * m / t:>9.3f}%"
                      f"{ws[len(ws) // 2]:>12.2e}{us[len(us) // 2]:>12.2e}")
    del rows
    gc.collect()


if __name__ == "__main__":
    main()
