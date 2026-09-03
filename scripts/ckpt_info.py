#!/usr/bin/env python3
# restartable: reads checkpoint metadata (mmap where available) and writes nothing, one
# JSON line per path to stdout. An interrupt loses at most the current file's read; rerun
# the same command. The one non-metadata read is torch.equal on same-shape weight pairs to
# detect tying, measured at 0.01s for two bf16 32832x1024 tensors.
"""One-line summary for a checkpoint: params, dtype, step, arch.

    python scripts/ckpt_info.py ckpt_k4_11b_lr05.pt [more.pt ...]

Prints one JSON object per line (JSONL): path, exists, step, params, and the
arch fields the CLI renders (layers, d, heads, attn_res), plus file size + mtime.
Reads only checkpoint metadata (mmap when available), so it stays cheap on the
~400MB checkpoints. Never raises: a bad file yields {"path", "error"}.
"""

import json
import os
import sys
from datetime import UTC, datetime


def count_params(model):
    """(distinct params, sum over state_dict entries) for a state_dict.

    A TIED weight appears under two keys pointing at ONE tensor in the live model:
    model.py:379 does `self.head.weight = self.tok.weight` unless untie_head. Summing
    entries counts it twice, and the wrong total is not obviously wrong -- at d1024 it
    lands on 206,128,200 + 32832*1024 = 239,748,168, which is the REAL parameter count
    of the --untie_head arm (EXPERIMENTS.md:223). A bug whose output is another
    experiment's true value does not read as a bug: it was taken for an "includes
    embeddings" convention and nearly reopened a launched leg's equal-FLOPs balance.

    Dedupe by VALUE, not by storage. torch.save serialises a tied weight as two
    independent copies -- MEASURED on ckpt_data_leg_206m_8b.pt.step4500 (untie_head
    false): tok.weight and head.weight come back with different data_ptr under both
    mmap=True and mmap=False, so a storage/data_ptr key finds nothing to dedupe and
    reports 239,748,168 for a tied checkpoint. Sharing exists in the model, not in the
    file.

    Only same-shape pairs are compared, torch.equal on the two bf16 32832x1024 tensors
    costs 0.01s, and the discrimination is verified on real files both ways:
      ckpt_data_leg_206m_8b.pt.step4500       untie_head=False -> 206,128,200
      ckpt_ab_untieheadlr_untieheadlr.pt.500  untie_head=True  -> 239,748,168
    Both numbers are returned because their difference IS the signal: nonzero means
    tied weights, which is otherwise invisible here (cfg carries untie_head only on
    runs new enough to have the field, and this file is pointed at old checkpoints).
    """
    ts = [t for t in model.values() if hasattr(t, "numel") and hasattr(t, "shape")]
    total = sum(int(t.numel()) for t in ts)
    kept = []  # tensors whose values are distinct from everything already counted
    distinct = 0
    for t in ts:
        dup = False
        for k in kept:
            if k.shape == t.shape and k.dtype == t.dtype:
                try:
                    dup = bool(_equal(k, t))
                except Exception:
                    dup = False  # unequal-by-failure: over-count rather than under-count
                if dup:
                    break
        if not dup:
            kept.append(t)
            distinct += int(t.numel())
    return distinct, total


def _equal(a, b):
    import torch

    return torch.equal(a, b)


def load_meta(path):
    import torch

    # mmap=True reads numel/cfg as metadata without pulling ~400MB off disk; fall
    # back if the runtime rejects it.
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except Exception:
        ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck.get("cfg", {}) if isinstance(ck, dict) else {}
    model = ck.get("model") if isinstance(ck, dict) else None
    params, sd_sum = None, None
    if isinstance(model, dict):
        try:
            params, sd_sum = count_params(model)
        except Exception:
            params, sd_sum = None, None
    return {
        "step": ck.get("step") if isinstance(ck, dict) else None,
        "params": params,
        "params_state_dict_sum": sd_sum,
        "tied": None if params is None else params != sd_sum,
        "layers": cfg.get("layers"),
        "d": cfg.get("d"),
        "heads": cfg.get("heads"),
        "attn_res": cfg.get("attn_res"),
    }


def info(path):
    rec = {"path": path, "exists": os.path.exists(path)}
    if not rec["exists"]:
        return rec
    st = os.stat(path)
    rec["size_bytes"] = st.st_size
    rec["date"] = datetime.fromtimestamp(st.st_mtime, tz=UTC).strftime("%Y-%m-%d")
    try:
        rec.update(load_meta(path))
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"
    return rec


def selftest():
    """Prove count_params separates tied from untied. Run: ckpt_info.py --selftest

    Every fixture goes through the same per-key copy the real save site does. An
    in-memory dict is NOT a valid fixture here: `{"tok.weight": t, "head.weight": t}`
    shares storage, so a storage-keyed dedupe passes on it and still reports
    239,748,168 on the checkpoint on disk -- which is what the first version of this fix
    did. train.py:2418 saves `{k: v.cpu() for k, v in state_dict().items()}`, and .cpu()
    on an already-CPU tensor returns the same object while a real GPU->CPU copy does
    not: the two keys become two independent tensors, which is why the measured
    checkpoint has no sharing left to find. `.clone()` per key reproduces that.
    """
    import io

    import torch

    d, pv = 8, 64
    emb, blk = pv * d, d * d

    def roundtrip(sd):
        # .clone() per key = train.py:2418's per-key .cpu() off GPU: sharing does not survive.
        buf = io.BytesIO()
        torch.save({"model": {k: v.clone() if hasattr(v, "clone") else v for k, v in sd.items()}}, buf)
        buf.seek(0)
        return torch.load(buf, map_location="cpu", weights_only=False)["model"]

    tok = torch.randn(pv, d)
    tied_live = {"tok.weight": tok, "head.weight": tok, "blk.w": torch.randn(d, d)}
    untied_live = {"tok.weight": tok, "head.weight": torch.randn(pv, d), "blk.w": torch.randn(d, d)}
    tied, untied = roundtrip(tied_live), roundtrip(untied_live)
    fails = []

    # 0. The fixture itself must be adversarial: the round trip must have BROKEN sharing,
    # or this file is testing the easy case and the real checkpoint is untested.
    if tied["tok.weight"].untyped_storage().data_ptr() == tied["head.weight"].untyped_storage().data_ptr():
        fails.append("fixture is not adversarial: saved tied weights still share storage here, "
                     "so this selftest cannot see the bug the real checkpoints have")

    got = count_params(tied)
    if got != (emb + blk, 2 * emb + blk):
        fails.append(f"tied: want {(emb + blk, 2 * emb + blk)}, got {got}")
    got = count_params(untied)
    if got != (2 * emb + blk, 2 * emb + blk):
        fails.append(f"untied: want {(2 * emb + blk, 2 * emb + blk)}, got {got}")
    if count_params(tied)[0] == count_params(untied)[0]:
        fails.append("tied and untied give the SAME distinct count: the dedupe is not running")
    # Two DIFFERENT tensors that happen to share a shape must not collapse.
    two = roundtrip({"a": torch.randn(pv, d), "b": torch.randn(pv, d)})
    if count_params(two)[0] != 2 * emb:
        fails.append("two distinct same-shape tensors were deduped: comparison is not by value")
    # numel-less values must not crash the count (real checkpoints carry ints and strings).
    if count_params({"tok.weight": tok, "step": 2000, "note": "x"}) != (emb, emb):
        fails.append("non-tensor entries changed the count")

    for f in fails:
        print("FAIL " + f)
    print(f"selftest: {len(fails)} FAIL of 6 cases")
    return 1 if fails else 0


def main(paths):
    if "--selftest" in paths:
        return selftest()
    for p in paths:
        print(json.dumps(info(p), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
