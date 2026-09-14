"""Pairing-contract test for eval/sampling.py (stage-2, 3b changes-requested 2026-09-14).

Contract: T and C checkpoints at the same (task_id, sample_idx) must draw from
the SAME RNG state at step 0, even when an earlier sample diverged in length
between the two runs and consumed a different number of multinomial scalars.
That holds only with a reseed PER (task_id, si); a single task-level seed
misaligns si>=1.

  python3 eval/test_sampling_paired.py   # from repo root: python eval/test...
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import sampling

V = 100_000
TEMPERATURE = 0.2
PROMPT = [7]
MAX_NEW = 8
TASK = "HumanEval/0"
N = 4


class FakeModel:
    """Uniform over non-special tokens until gen>=stop_after, then eos.

    Two instances with different stop_after reproduce differing sample-0
    lengths between T and C (and therefore differing RNG scalar counts).
    """

    def __init__(self, stop_after):
        self.stop_after = stop_after

    def __call__(self, x):
        gen = x.shape[1] - len(PROMPT)
        logits = torch.full((1, x.shape[1], V), -1e9)
        col = logits[:, -1]
        if gen >= self.stop_after:
            col[:, sampling.EOS_TID] = 100.0
        else:
            col[:, 2:] = 0.0
        return (logits,)


class FakeTok:
    def decode(self, ids):
        return str(list(ids))


def first_draws(stop_after, reseed_per_si):
    """First multinomial token of each of the n samples for one fake run.

    Each sample consumes stop_after+1 draws (stop_after tokens then eos), so
    with known offsets the first draw per sample is read back from the log.
    """
    log = []
    real = torch.multinomial

    def wrapped(probs, k, *a, **kw):
        r = real(probs, k, *a, **kw)
        log.append(int(r.item()))
        return r

    torch.multinomial = wrapped
    try:
        if reseed_per_si:
            sampling.sample_completions(
                FakeModel(stop_after), FakeTok(), PROMPT, TASK, N,
                TEMPERATURE, MAX_NEW, "cpu", 64)
        else:
            _once_seeded(stop_after)
    finally:
        torch.multinomial = real
    width = stop_after + 1
    return [log[si * width] for si in range(N)]


@torch.no_grad()
def _once_seeded(stop_after):
    """The old contract: one task-level seed before the n-sample loop."""
    x0 = torch.tensor([PROMPT])
    torch.manual_seed(sampling.task_seed(TASK))
    for _ in range(N):
        x = x0
        for _ in range(MAX_NEW):
            logits = FakeModel(stop_after)(x[:, -64:])[0][:, -1]
            nxt = torch.multinomial(torch.softmax(logits.float() / TEMPERATURE, -1), 1)
            if nxt.item() == sampling.EOS_TID:
                break
            x = torch.cat([x, nxt], 1)


def main():
    # 1. THE FIX: differing sample-0 lengths (3 vs 2 scalars) must not misalign
    #    si>=1, because every sample reseeds from (task_id, si).
    t = first_draws(stop_after=2, reseed_per_si=True)
    c = first_draws(stop_after=1, reseed_per_si=True)
    assert t == c, f"per-si reseed failed to align streams: T={t} C={c}"

    # 2. THE DEFECT as a discriminating counterexample: one seed per task leaves
    #    si>=1 at different offsets; the wide-uniform first draw then differs
    #    (collision chance 1/V).
    t0 = first_draws(stop_after=2, reseed_per_si=False)
    c0 = first_draws(stop_after=1, reseed_per_si=False)
    assert t0[0] == c0[0], "sample 0 must align even under the old scheme"
    assert any(t0[i] != c0[i] for i in range(1, N)), (
        "reseed-once stayed aligned after differing lengths -- counterexample "
        "no longer discriminates")

    # 3. Distinct si seed differently: first draws are not all one token.
    assert len(set(t)) > 1, "all si drew the same token -- seeds not distinct"

    print(f"sampling pairing OK: per-si T={t} == C={c}; reseed-once misaligns "
          f"si>=1 (T={t0}, C={c0})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
