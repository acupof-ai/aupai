#!/usr/bin/env python3
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
    params = None
    if isinstance(model, dict):
        try:
            params = sum(int(t.numel()) for t in model.values() if hasattr(t, "numel"))
        except Exception:
            params = None
    return {
        "step": ck.get("step") if isinstance(ck, dict) else None,
        "params": params,
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


def main(paths):
    for p in paths:
        print(json.dumps(info(p), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
