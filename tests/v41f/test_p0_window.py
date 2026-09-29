"""P0: sliding-window indices match upstream (prefill causal + decode ring)."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).absolute().parent))
from ref_oracle import load_reference

sys.path.insert(0, str(Path(__file__).absolute().parents[2]))
from v41f.window import get_window_topk_idxs


def test_prefill_short():
    model, _ = load_reference()
    win, s = 8, 5  # seq shorter than window: early rows carry -1 empties
    got = get_window_topk_idxs(win, 1, s, 0)
    want = model.get_window_topk_idxs(win, 1, s, 0)
    assert torch.equal(got, want), (got[0].tolist(), want[0].tolist())


def test_prefill_long():
    model, _ = load_reference()
    win, s = 4, 10  # wraps: every row sees exactly win slots, causal
    got = get_window_topk_idxs(win, 2, s, 0)
    want = model.get_window_topk_idxs(win, 2, s, 0)
    assert torch.equal(got, want)
    # row t sees max(t,0)..min(t,win-1) real slots, all <= t, no future
    real = got[0][got[0] >= 0]
    assert real.max().item() <= s - 1
    # last row covers exactly the trailing window
    last = got[0, -1]
    assert set(last[last >= 0].tolist()) == {6, 7, 8, 9}


def test_decode_ring():
    model, _ = load_reference()
    win = 4
    for sp in (1, 4, 7):
        got = get_window_topk_idxs(win, 1, 1, sp)
        want = model.get_window_topk_idxs(win, 1, 1, sp)
        assert torch.equal(got, want), (sp, got.tolist(), want.tolist())
