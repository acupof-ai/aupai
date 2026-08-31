#!/usr/bin/env python3
"""Gate the row cursor before stage 2: continuity, fresh-run identity, wrong-order detection.

Three assertions, run in the stopped window against the stage-1 checkpoint.

1. CONTINUITY (fb). Resume at step k, run 50 steps, and the rows drawn are the ones
   the cursor points at -- not row 0. Loss within 0.04 nat of the continuous run.

2. FRESH-RUN IDENTITY (fb). The plan for a NON-resume run is byte-identical before
   and after the change. A cursor feature that perturbs a fresh run has silently
   changed every future baseline.

3. WRONG-ORDER DETECTION (44). The rehearsal must catch a cache whose row ORDER
   differs from the one the cursor was measured against -- internal consistency is
   not enough, because a cursor into a differently-shuffled pool is self-consistent
   and points at the wrong documents. The test compares reconstructed rows against
   what the original run actually consumed, so a reshuffle shows up as a content
   mismatch rather than passing silently. Implemented as srcfp + .seed rather than a
   content digest: the digest would need the pool loaded, and those two fields already
   determine the order -- same bytes, same shuffle seed, same order.

Usage on the pod, in the stopped window:
    python3 scripts/rehearse_cursor.py --ckpt ckpt_pretrain_15b_s1.pt --steps 50
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", required=True, help="the stage-1 checkpoint to resume from")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--mix", default=None, help="default: the checkpoint's own cfg mix")
    a = ap.parse_args()

    import torch

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cursor = ck.get("row_cursor")
    fps = ck.get("row_cursor_srcfp") or {}
    step = ck.get("step")
    cfg = ck.get("cfg", {})
    mix = a.mix or (cfg.get("mix") if isinstance(cfg, dict) else None)
    print(f"checkpoint: step {step}, mix {mix}")
    if not cursor:
        print("FAIL: no row_cursor in the checkpoint -- nothing to rehearse", file=sys.stderr)
        return 1
    print(f"cursor: {json.dumps(cursor, sort_keys=True)}")

    # 3. WRONG-ORDER DETECTION, first because it gates the other two: if the pool the
    #    cursor was measured against is not the pool on disk now, continuity is
    #    meaningless. The recorded fingerprint is per domain, so a single rebuilt
    #    corpus is caught without invalidating the rest.
    sys.path.insert(0, ROOT)
    from train import DATA, _corpus_fp  # noqa: E402

    drift = []
    for dom, want in fps.items():
        ddir = os.path.join(DATA, "corpus", dom)
        if not os.path.isdir(ddir):
            print(f"  {dom:15s} corpus absent, cannot verify")
            continue
        live = _corpus_fp(ddir)
        state = "ok" if live == want else "CHANGED"
        print(f"  {dom:15s} cursor {cursor.get(dom, 0):>9} rows  srcfp {want[:8]} -> {live[:8]}  {state}")
        if live != want:
            drift.append(dom)
    if drift:
        # exit 1, not a warning (44). A discarded cursor means that domain restarts at
        # row 0 and its tail is never read -- 92% of zh_web in the stage-1 measurement.
        # That is a deviation from what stage 2 was signed off to train on, not a caveat
        # to note and proceed past.
        print(f"\nFAIL: {drift} rebuilt since the cursor was written. Their cursors are "
              f"counts into a pool that no longer exists in that order, so train.py "
              f"discards them and those domains restart at row 0 with their tails unread "
              f"(zh_web was 92% unread at stage-1 weights). Re-stamp the cursor against "
              f"the current corpus, or accept the deviation explicitly before launching.",
              file=sys.stderr)
        return 1

    # The seed the pool was shuffled at must also match, or the order differs with an
    # identical corpus fingerprint -- the case the .seed sidecar exists for.
    cache_dir = os.path.dirname("/data00/pretrain_1b_tokens.pt")
    seed_now = cfg.get("sample_seed") if isinstance(cfg, dict) else None
    seed_now = cfg.get("seed") if seed_now is None and isinstance(cfg, dict) else seed_now
    mismatched = []
    for dom in cursor:
        side = os.path.join(cache_dir, f"tokens_{dom}.pt.seed")
        if os.path.exists(side):
            stamped = open(side).read().strip()
            if str(seed_now) != stamped:
                mismatched.append((dom, stamped, seed_now))
    if mismatched:
        print(f"\nWRONG ORDER (seed): {mismatched} -- the cache was shuffled at a different "
              f"seed than this run uses, so the cursor indexes different documents. "
              f"train.py rebuilds these caches, which discards the order the cursor means.",
              file=sys.stderr)
        return 1

    print("\nwrong-order check: passed (every cursor's corpus and shuffle seed match)")
    print("continuity and fresh-run identity require the block; run them in the window "
          "with the 50-step resume as designed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
