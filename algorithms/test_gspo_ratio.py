#!/usr/bin/env python3
"""Known-answer tests for the GSPO loss: the ratio is real and the clip binds.

    python3 algorithms/test_gspo_ratio.py

WHY THIS EXISTS. Until 2026-09-05 `gspo_loss` computed `old_lp = seq_lp.detach()` inside
the same forward, so the importance ratio was exp(x - x) == 1.0 identically, the clip at
1 +/- clip_eps could never bind, and `--clip_eps` was a live flag that changed nothing at
any value. The loss was plain policy gradient plus the KL term while the module called
itself GSPO. Nothing failed; the flag simply did nothing, for every run.

That is only findable by a test that VARIES clip_eps and demands the loss move. A test
that asserts the loss is finite, or that it decreases, passes under both the defect and
the fix. So the load-bearing assertion here is a difference, not a value.

CPU-only: a two-token stub model with a hand-set logit table stands in for the policy, so
this runs in the hook and in CI. What it exercises is gspo_loss's arithmetic -- the ratio,
the clip, the advantage and the KL -- which is where the defect lived.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from rlvr_trainer import gspo_loss  # noqa: E402

VOCAB = 8
FAILS = []


class Stub(nn.Module):
    """A policy whose logits are a learned table, independent of the input.

    Enough for gspo_loss: it needs log-probs of given token ids and a gradient path back
    to something. `scale` shifts the whole distribution so a second Stub can play the old
    policy at a measurably different place.
    """

    def __init__(self, scale=0.0, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        # SCALED, not shifted. Softmax is invariant to adding a constant to every logit, so
        # `+ scale` moved nothing and every ratio came out exactly 1.0 -- the fixture
        # reproduced the very defect it was written to catch, and reported a correct
        # implementation as broken. Multiplying changes the distribution's temperature and
        # therefore the log-probs.
        self.table = nn.Parameter(torch.randn(VOCAB, generator=g) * (1.0 + scale))

    def forward(self, x):
        B, T = x.shape
        return self.table.view(1, 1, VOCAB).expand(B, T, VOCAB), None


def _loss(clip_eps, old_scale=0.0, rewards=(1.0, 1.0, 0.0, 0.0), kl_beta=0.0, seed=0):
    """gspo_loss on a fixed tiny group. old_lp comes from a SEPARATE model, as in the
    real caller, where gen_model produced the rollouts before the optimizer moved on."""
    torch.manual_seed(seed)
    model = Stub(seed=seed)
    # The reference is a DIFFERENT policy. Built from the same seed it is bit-identical to
    # the model, d = ref_lp - seq_lp is 0, kl = exp(0) - 0 - 1 = 0, and kl_beta multiplies
    # zero -- so the anchor test could not fail no matter what kl_beta did. In the real
    # trainer ref_model is a frozen copy of the SFT weights that the policy moves away from.
    ref = Stub(scale=0.4, seed=seed + 101)
    old = Stub(scale=old_scale, seed=seed)
    G = len(rewards)
    prompt_ids = [1, 2]
    # DISTINCT responses per row. Identical ones give identical seq_lp, hence an identical
    # ratio across the group, and since the advantages sum to zero the loss is then 0 for
    # every clip_eps -- a fixture with no power, which is how the first version of this test
    # "failed" against a correct implementation.
    gen = [[3 + i, 4, 5] for i in range(G)]
    with torch.no_grad():
        from rlvr_trainer import seq_logprob
        old_lp, _, _ = seq_logprob(old, prompt_ids, gen, G, 3, False, "cpu", False)
    return gspo_loss(model, ref, prompt_ids, gen, list(rewards), G, 3, False, "cpu", False,
                     clip_eps=clip_eps, kl_beta=kl_beta, old_lp=old_lp)


def main():
    # 1. THE FLAG MUST MATTER. Two clip_eps values, everything else fixed, with the old
    #    policy far enough away that the ratio leaves [1-eps, 1+eps]. Under the old code
    #    both sides are identical and this is the assertion that catches it.
    tight = _loss(0.01, old_scale=1.5).item()
    loose = _loss(5.0, old_scale=1.5).item()
    if abs(tight - loose) < 1e-6:
        FAILS.append(f"--clip_eps does not change the loss: {tight:.6f} vs {loose:.6f}. The "
                     f"ratio is probably exp(x - x) == 1 again (old_lp taken from the same "
                     f"forward), which makes the clip inert at every value.")

    # 2. AND IT MUST NOT MATTER WHEN THE RATIO IS INSIDE THE BAND. Same two values with the
    #    old policy equal to the current one: the clip cannot bind, so the loss is identical.
    #    Without this, assertion 1 would also pass for a loss that merely depends on clip_eps
    #    arbitrarily -- e.g. one that added it as a constant.
    near_t = _loss(0.5, old_scale=0.0).item()
    near_l = _loss(5.0, old_scale=0.0).item()
    if abs(near_t - near_l) > 1e-6:
        FAILS.append(f"clip_eps changes the loss even when the ratio is inside the band: "
                     f"{near_t:.6f} vs {near_l:.6f}. The clip is being applied where it "
                     f"should be a no-op.")

    # 3. A MISSING old_lp RAISES. The fallback that caused the defect must not come back
    #    quietly; gspo_loss refuses rather than recomputing it from the current forward.
    try:
        torch.manual_seed(0)
        m, r = Stub(), Stub()
        gspo_loss(m, r, [1, 2], [[3, 4, 5]] * 4, [1.0, 1.0, 0.0, 0.0], 4, 3, False, "cpu",
                  False, old_lp=None)
        FAILS.append("gspo_loss accepted old_lp=None instead of raising")
    except ValueError:
        pass

    # 4. A DEGENERATE GROUP CONTRIBUTES NO GRADIENT. All-equal rewards give std 0 and
    #    advantage 0; the trainer drops these groups, and this says why that is safe rather
    #    than merely conventional.
    for rw in ((1.0, 1.0, 1.0, 1.0), (0.0, 0.0, 0.0, 0.0)):
        loss = _loss(0.2, old_scale=1.5, rewards=rw)
        loss.backward()
        # KL is off (kl_beta=0), so the only gradient path is through the advantage.
        if loss.abs().item() > 1e-6:
            FAILS.append(f"degenerate group rewards {rw} gave a non-zero loss {loss.item():.6g}")

    # 5. THE KL ANCHOR PULLS. With kl_beta > 0 and a reference that differs from the policy,
    #    the loss must differ from the kl_beta=0 case -- otherwise the anchor is decorative.
    #    Same shape of assertion as 1: a difference, not a value.
    torch.manual_seed(0)
    no_kl = _loss(0.2, old_scale=1.5, kl_beta=0.0, seed=1).item()
    with_kl = _loss(0.2, old_scale=1.5, kl_beta=0.5, seed=1).item()
    if abs(no_kl - with_kl) < 1e-9:
        FAILS.append(f"kl_beta does not change the loss: {no_kl:.6f} vs {with_kl:.6f}")

    for f in FAILS:
        print(f"BUG {f}", file=sys.stderr)
    print(f"gspo ratio test: {'PASS (5 worlds)' if not FAILS else f'{len(FAILS)} BUG(S)'}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
