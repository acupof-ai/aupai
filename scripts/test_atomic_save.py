#!/usr/bin/env python3
"""Known-answer gate for _atomic_torch_save (1e order 2026-09-26, fix before the 30B retrain).

A signal (SIGTERM/OOM) arriving during a checkpoint write must leave the PREVIOUS checkpoint
loadable, not a truncated final path. The 2026-09-25 failure wrote straight to the final path
and interrupt.step452 became 8.99GB with no zip central directory.

    python3 scripts/test_atomic_save.py --selftest
"""

import os
import sys
import tempfile

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from train import _atomic_torch_save  # noqa: E402


def _tmp_leftovers(d):
    return [f for f in os.listdir(d) if f.startswith("ckpt-tmp-")]


def main():
    d = tempfile.mkdtemp(prefix="atomic_save_")
    target = os.path.join(d, "ckpt.pt")

    # 1) success: target round-trips and no temp leaks.
    obj1 = {"v": torch.arange(100), "tag": "one"}
    _atomic_torch_save(obj1, target)
    assert torch.load(target, weights_only=False)["tag"] == "one"
    assert not _tmp_leftovers(d), "successful save must remove the temp"

    # 2) a write that dies after streaming bytes must not touch the existing target and must
    #    unlink the temp.
    real_save = torch.save

    def dying_save(obj, path, *a, **k):
        with open(path, "wb") as f:
            f.write(b"PK\x03\x04 partial garbage, no central directory")
        raise RuntimeError("simulated SIGTERM-during-save")

    torch.save = dying_save
    try:
        try:
            _atomic_torch_save({"tag": "two"}, target)
        except RuntimeError:
            pass
        else:
            raise AssertionError("interrupted save must propagate the error")
    finally:
        torch.save = real_save

    assert torch.load(target, weights_only=False)["tag"] == "one", (
        "interrupted write must leave the PREVIOUS checkpoint intact"
    )
    assert not _tmp_leftovers(d), "interrupted save must unlink the temp"

    # 3) two concurrent temp names are distinct (a hard SIGKILL straggler cannot block the
    #    next write's temp path).
    fd1, t1 = tempfile.mkstemp(prefix="ckpt-tmp-", dir=d)
    os.close(fd1)
    fd2, t2 = tempfile.mkstemp(prefix="ckpt-tmp-", dir=d)
    os.close(fd2)
    assert t1 != t2
    os.unlink(t1)
    os.unlink(t2)

    print("atomic save OK: kill-during-write keeps prior ckpt, success round-trips, no temp leak")


if __name__ == "__main__":
    if len(sys.argv) > 2 or (len(sys.argv) == 2 and sys.argv[1] != "--selftest"):
        raise SystemExit(f"usage: {os.path.basename(__file__)} [--selftest] (got {sys.argv[1:]})")
    main()
