#!/usr/bin/env python3
"""The rollout policy and the scored policy must be conditioned on the SAME context.

algorithms/rlvr_generate.generate samples the next token; algorithms/rlvr_trainer.seq_logprob
scores the sampled sequence in one forward over cat([prompt, gen]). GSPO's importance ratio
exp(seq_lp - old_lp) is only a ratio between two policies if both were evaluated on the same
conditioning prefix. Until 2026-09-30 generate read `model(x[:, -1024:])` -- a fixed window --
while seq_logprob read the whole sequence, so at any prompt+completion longer than 1024 tokens
the two sides computed different conditionals and the clip bounded the wrong quantity. The
stdin code shape is MAX_PROMPT_STDIN 1024 + MAX_NEW_STDIN 562 = 1586, i.e. truncated from the
first generated token on.

The assertion is on the CALL SITES, not on a re-derivation of the slicing rule: one spy module
serves both functions and records the exact tensor each was handed. Step t of generation must
have seen the scorer's input truncated at plen+t, byte for byte, and the scorer's own seq_lp
must equal the mean of the generation-time token log-probs.

The negative control is the defect itself: ctx_window=1024 (what the line used to do) with a
prompt long enough to trigger it must fail both arms. A check that only ran the fixed path
could pass by not exercising the window at all.

    python3 algorithms/test_rollout_context_parity.py            # same as --selftest
    python3 algorithms/test_rollout_context_parity.py --selftest
"""

import os
import sys

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from rlvr_generate import generate  # noqa: E402
from rlvr_trainer import seq_logprob  # noqa: E402

VOCAB = 37
DIM = 8
EOS = 1


class SpyLM(nn.Module):
    """A causal toy LM that records every input it is handed.

    cumsum over the embedded prefix makes position t depend on the WHOLE prefix, so a
    truncated context produces different logits -- without that, a windowing defect would be
    numerically invisible and only the recorded shapes would catch it.
    """

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.emb = nn.Embedding(VOCAB, DIM)
        self.out = nn.Linear(DIM, VOCAB)
        self.seen = []

    def forward(self, x, targets=None, cu=None, num_vals=None, no_head=False):
        self.seen.append(x.detach().clone())
        h = self.emb(x).cumsum(dim=1) / torch.arange(
            1, x.size(1) + 1, device=x.device, dtype=self.emb.weight.dtype
        ).view(1, -1, 1)
        return self.out(h), None


def _run(prompt_len, max_new, ctx_window):
    """(model, prompt_ids, gen_ids, gen_inputs, score_input, seq_lp) for one rollout+score."""
    torch.manual_seed(1234)
    model = SpyLM()
    # ids 2.. only: EOS=1 would end a row early and shorten the arms under test
    prompt_ids = [2 + (i * 7) % (VOCAB - 2) for i in range(prompt_len)]
    with torch.no_grad():
        gen = generate(model, prompt_ids, 1, max_new, 1.0, 1.0, "cpu", ctx_window=ctx_window)
    gen_inputs = list(model.seen)
    model.seen.clear()
    # Rows that stopped at <eos> make the two sides disagree about length for a reason that is
    # not the context; the toy sampler is unconstrained, so just require a full-length row.
    assert len(gen[0]) == max_new, f"toy rollout ended early ({len(gen[0])}); rerun with a seed that does not"
    with torch.no_grad():
        seq_lp, _, _ = seq_logprob(model, prompt_ids, gen, 1, max_new, False, "cpu", False)
    score_inputs = [t for t in model.seen]
    assert len(score_inputs) == 1, f"seq_logprob made {len(score_inputs)} forwards, expected 1"
    return model, prompt_ids, gen[0], gen_inputs, score_inputs[0], float(seq_lp[0])


def _parity(prompt_len, max_new, ctx_window):
    """(prefixes_match, seq_lp_matches) for one configuration. Never raises on a mismatch --
    the caller decides which outcome is the expected one."""
    model, prompt_ids, gen, gen_inputs, score_input, seq_lp = _run(prompt_len, max_new, ctx_window)
    plen = len(prompt_ids)
    assert len(gen_inputs) == max_new, f"{len(gen_inputs)} generation forwards for max_new={max_new}"
    prefixes_match = all(
        torch.equal(gen_inputs[t], score_input[:, : plen + t]) for t in range(max_new)
    )
    # The scorer's seq_lp is the mean token log-prob; recompute it from the tensors GENERATION
    # actually saw, so the two numbers can only agree if both sides used one context.
    lps = []
    with torch.no_grad():
        for t in range(max_new):
            logits, _ = model(gen_inputs[t])
            lps.append(float(torch.log_softmax(logits[0, -1, :].float(), dim=-1)[gen[t]]))
    gen_mean = sum(lps) / len(lps)
    return prefixes_match, abs(gen_mean - seq_lp) < 1e-4


def selftest():
    # The shape the defect bit on: 1024-token prompt + 562 new is the stdin code arm
    # (rl_code_trainer.MAX_PROMPT_STDIN / MAX_NEW_STDIN). Scaled down to keep this a
    # sub-second CPU test; what matters is that plen exceeds the window under test.
    plen, max_new, window = 40, 12, 32
    ok_prefix, ok_lp = _parity(plen, max_new, ctx_window=None)
    assert ok_prefix, "generate did not condition on the scorer's own prefix at every step"
    assert ok_lp, "seq_logprob's seq_lp does not equal the mean generation-time token log-prob"
    print(f"  ctx_window=None (the trainers' call): prefixes identical at all {max_new} steps, "
          f"seq_lp == generation mean")

    # NEGATIVE CONTROL: the pre-fix line. It must break BOTH arms, or neither arm is measuring
    # the window.
    bad_prefix, bad_lp = _parity(plen, max_new, ctx_window=window)
    assert not bad_prefix, (
        f"ctx_window={window} with a {plen}-token prompt did not change the conditioning "
        f"prefix -- the negative control never exercised the window, so the positive arm "
        f"proves nothing")
    assert not bad_lp, (
        f"ctx_window={window} left seq_lp equal to the generation mean; the toy model is not "
        f"context-sensitive and the numeric arm is inert")
    print(f"  ctx_window={window} (the 2026-09-30 defect): prefixes differ AND seq_lp diverges")

    # No caller may pass ctx_window: it exists only for the control above.
    import glob
    import re

    offenders = []
    # Excluding this file by its own name is not enough: the commit hook runs the STAGED blob as
    # .hookstaged_<name>.py, so __file__ no longer matches and the scan finds its own control
    # fixtures. Match the suffix, which both spellings share.
    me = os.path.basename(__file__).lstrip(".").removeprefix("hookstaged_")
    for path in glob.glob(os.path.join(HERE, "*.py")):
        # rlvr_generate.py DEFINES the parameter; every other module may only not pass it.
        base = os.path.basename(path)
        if base.endswith(me) or base == "rlvr_generate.py":
            continue
        for i, line in enumerate(open(path, encoding="utf-8"), 1):
            if re.search(r"\bctx_window\s*=", line):
                offenders.append(f"{os.path.relpath(path)}:{i}")
    assert not offenders, f"ctx_window passed by a real call site: {offenders}"
    print("  no algorithms/ call site passes ctx_window")
    print("rollout context parity selftest OK")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] not in ("--selftest",):
        sys.exit(f"usage: {sys.argv[0]} [--selftest]")
    selftest()
