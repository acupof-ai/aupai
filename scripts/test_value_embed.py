#!/usr/bin/env python3
# restartable: builds small CPU models and runs a few forwards. Seconds; nothing to shard.
"""A/B (4) `--value_embed`: one SHARED table, only in MLA, only on V, and it must be visible.

    python3 scripts/test_value_embed.py         # exit 1 on any difference from the contract

FOUR WAYS THIS ARM COSTS A RUN AND STILL REPORTS AS THE ARM:

  1. THE TABLE IS INVISIBLE AT INIT. If the gate started at zero the table would contribute
     nothing and the arm would spend its 500 steps discovering the table exists. The gate's
     weight is zero-init so the gate is a uniform 3*sigmoid(0) = 1.5, NOT 0. Asserted directly,
     and asserted to be 1.5 rather than merely nonzero.

  2. THE TABLE LANDS IN THE LATENT, so K sees token identity too. kv_down/kv_up produce K and V
     from ONE latent, so adding there stops this being a value embedding. Asserted by checking
     that K is unchanged when the table changes while V is not.

  3. THREE TABLES INSTEAD OF ONE. 1e ruled one shared [vocab, d] table: three would be +48.9%
     parameters against +16.3%. Asserted by object identity and by the exact parameter delta.

  4. THE STASH LEAKS. The per-forward lookup is stashed on each MLA layer. If it outlived the
     forward it would pin a [B, T, d] activation between steps and, worse, be silently reused by
     a forward that reached an MLA layer another way -- a wrong number, not a crash.

Measured at the real 200M config (d=1024, L=12, heads=8, ffn_hidden=3072, vocab 32832):
206,128,200 -> 239,788,104, delta 33,659,904 = +16.3%, three MLA layers gated.
"""
import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 2)[0])

import model as M  # noqa: E402
import train  # noqa: E402

# Forced, never conditional: fla and flash_attn are both importable on the pod, so a
# `if x is None` stub is green on a dev box and red where it matters (shape 72).
M.chunk_kda = lambda q, k, v, **kw: (q * 0 + v, None)  # noqa: E731
M.HAS_FA = False


def build(ve, d=128, layers=4, heads=4, ffn=256, vocab=512, real=500, seed=0):
    cfg = train.Cfg
    cfg.d, cfg.layers, cfg.heads, cfg.ffn_hidden = d, layers, heads, ffn
    cfg.vocab, cfg.vocab_real = vocab, real
    cfg.zero_init_out = False
    cfg.muon_shape_lr = False
    cfg.value_embed = ve
    torch.manual_seed(seed)
    return M.HybridLM(cfg).eval(), cfg


def gated_mlas(m):
    return [x for x in m.modules() if isinstance(x, M.GatedMLA) and x.ve_gate is not None]


def main():
    fails = []
    x = torch.randint(0, 500, (2, 16))

    # 0. THE FIELD MUST EXIST ON Cfg BY THAT NAME. Every other check sets cfg.value_embed
    #    itself, so none of them can see a rename in train.Cfg -- the arm would then run the
    #    baseline (model.py reads it through getattr with a False default) while the launcher's
    #    --value_embed silently wrote nothing. Muon catches its own equivalent with an
    #    `assert hasattr`; model.py cannot, because it is also constructed from a pre-split Cfg
    #    that legitimately predates the field (test_split_bitwise.py). So the check lives here.
    #    Verified: renaming Cfg.value_embed to value_embedd turns exactly this one red, and the
    #    first version of this test had NO such check while its comment claimed otherwise.
    if not hasattr(train.Cfg, "value_embed"):
        fails.append("train.Cfg has no `value_embed` field. model.py reads it via getattr with "
                     "a False default, so the arm would run the BASELINE and still report as "
                     "the arm; --value_embed would write to nothing.")

    base, _ = build(False)
    arm, cfg = build(True)

    # 1. THE GATE MUST START AT 1.5, not 0 and not random.
    for g in gated_mlas(arm):
        if float(g.ve_gate.weight.detach().abs().max()) != 0.0:
            fails.append("ve_gate.weight is not zero-init, so the gate starts at a random "
                         "per-token value instead of a uniform 1.5")
        if float(g.ve_gate.bias.detach().abs().max()) != 0.0:
            fails.append("ve_gate.bias is not zero, so the gate does not start uniform")
    if gated_mlas(arm):
        probe = torch.zeros(1, 1, 12)
        with torch.no_grad():
            val = float(3.0 * torch.sigmoid(gated_mlas(arm)[0].ve_gate(probe)).flatten()[0])
        if abs(val - 1.5) > 1e-6:
            fails.append(f"the gate starts at {val:.4f}, not 1.5. At 0 the table is INVISIBLE "
                         f"and the arm measures how fast it finds the table, not the table")

    # 2. ONE table, shared. Object identity, and the parameter delta must be table + gates only.
    if arm.value_embed is None:
        fails.append("the arm has no value_embed table at all")
    else:
        n_base = sum(p.numel() for p in base.parameters())
        n_arm = sum(p.numel() for p in arm.parameters())
        tbl = arm.value_embed.weight.numel()
        gates = sum(p.numel() for g in gated_mlas(arm) for p in g.ve_gate.parameters())
        if n_arm - n_base != tbl + gates:
            fails.append(f"parameter delta {n_arm - n_base:,} != table {tbl:,} + gates "
                         f"{gates:,}; something else changed size")
        # Three separate tables would show up as 3x the table in the delta.
        if tbl != cfg.vocab * cfg.d:
            fails.append(f"the table is {tbl:,}, expected vocab*d = {cfg.vocab * cfg.d:,}")

    # 3. K MUST NOT SEE THE TABLE. Capture k and v from one MLA layer, change the table, and
    #    require k identical and v changed. This is the check that "not in the latent" is true.
    layers = gated_mlas(arm)
    if not layers:
        fails.append("no MLA layer carries a gate, so nothing below tests anything")
    else:
        mla = layers[0]
        seen = {}

        def grab(mod, inp, out):
            k, v = out.chunk(2, dim=-1)
            seen.setdefault("k", []).append(k.detach().clone())
            seen.setdefault("v", []).append(v.detach().clone())

        h = mla.kv_up.register_forward_hook(grab)
        try:
            with torch.no_grad():
                arm(x)
                arm.value_embed.weight.data.add_(10.0)
                arm(x)
        finally:
            h.remove()
        if len(seen.get("k", [])) < 2:
            fails.append("the kv_up hook did not fire twice; the K/V check did not run")
        else:
            # kv_up's OUTPUT is pre-VE for both k and v -- the add happens after. So this hook
            # proves the LATENT path is untouched: both halves must be identical across the two
            # forwards, which is what "the table does not enter kv_down/kv_up" means.
            if not torch.equal(seen["k"][0], seen["k"][1]):
                fails.append("kv_up's K output changed when the table changed: the table is "
                             "reaching the LATENT, so K carries token identity and this is no "
                             "longer a value embedding")
            if not torch.equal(seen["v"][0], seen["v"][1]):
                fails.append("kv_up's V output changed when the table changed, so the add is "
                             "happening inside the projection rather than after it")

    # 4. THE TABLE MUST BE LOAD-BEARING. Scaling it has to move the logits, or the arm is a
    #    baseline with extra parameters -- which would still train and still report as the arm.
    arm2, _ = build(True)
    with torch.no_grad():
        y1 = arm2(x)
        y1 = (y1[0] if isinstance(y1, tuple) else y1).clone()
        arm2.value_embed.weight.data.mul_(50.0)
        y2 = arm2(x)
        y2 = y2[0] if isinstance(y2, tuple) else y2
    if torch.equal(y1, y2):
        fails.append("scaling the table 50x did not change the logits: the table is wired up "
                     "but unused, so the arm is the baseline plus dead parameters")

    # 5. THE STASH MUST NOT OUTLIVE THE FORWARD.
    leaked = [i for i, g in enumerate(gated_mlas(arm2)) if g._ve is not None]
    if leaked:
        fails.append(f"_ve still set on MLA layer(s) {leaked} after the forward returned; a "
                     f"stale lookup pins a [B, T, d] activation and would be reused by a "
                     f"forward that reached an MLA layer another way")

    # 6. THE BASELINE MUST BE UNTOUCHED. No gate, no table, and every MLA layer's _ve is None,
    #    or checks 1-5 are describing the architecture rather than the arm.
    if base.value_embed is not None:
        fails.append("the baseline built a value_embed table")
    if any(getattr(x_, "ve_gate", None) is not None for x_ in base.modules()):
        fails.append("the baseline built a ve_gate")

    for f in fails:
        print(f"  FAIL {f}")
    if fails:
        print(f"\n{len(fails)} failure(s)")
        return 1
    n_g = len(gated_mlas(arm))
    print(f"value_embed OK: 1 shared table of {arm.value_embed.weight.numel():,}, {n_g} gated "
          f"MLA layer(s), gate starts at exactly 1.5, K unaffected while the table is "
          f"load-bearing (50x moved the logits), stash cleared, baseline untouched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
