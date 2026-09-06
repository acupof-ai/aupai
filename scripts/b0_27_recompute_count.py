#!/usr/bin/env python3
"""b0-27 known-answer test: does checkpoint recompute double tokens_per_expert?

THE QUESTION THIS ANSWERS, and the one it does not. HybridLM.__init__ refuses moe_experts with
grad_ckpt, and its stated reason is that MoEFFN.forward increments tokens_per_expert under
no_grad while recompute runs that forward twice. That reasoning is plausible and has never been
observed: the condition guarding the increment is `self.training or torch.is_grad_enabled()`,
which I read as true in both passes, but READING A CONDITION IS NOT OBSERVING A COUNT. This
script observes it. Same fixed input, grad_ckpt off vs on, tokens_per_expert compared
elementwise. Ratio 1 means the guard has been refusing a configuration that was never broken;
ratio 2 on the MoE layers is the signature of the defect as described.

WHY A CARD IS REQUIRED, and it is not simply "the model uses Triton". MoEFFN itself has NO
Triton dependency -- the counter lives in a plain nn.Module, and grep for chunk_kda/l2norm/triton
inside `class MoEFFN` finds nothing. So the tempting move is to run the whole thing on CPU. That
does not work, and the reason is a refusal rather than a missing kernel: every valid config
carries at least one KDA layer, because HybridLM rejects attn_every=1 outright ("0 KDA layers,
but GatedMLA is NoPE ... the model would have no position information"). KDA needs chunk_kda,
which is None on a machine without the Triton build. The alternative -- stubbing chunk_kda, as
scripts/probe_gradckpt_sources.py does -- would answer the question on a DIFFERENT model, and
the whole point of this measurement is that reading a plausible mechanism is not observing it.
So: real model, real kernels, one card.

WHAT I ALREADY ESTABLISHED WITHOUT A CARD, so the finding is not overstated either way -- both
consumers of the counter are invariant to a UNIFORM 2x:
  * update_bias steps on torch.sign(counts - mean). Doubling every count doubles counts and
    mean together, err doubles, and sign() is scale-invariant, so the bias step is bit-identical
    (ties included). The zero-mean projection is not what saves it.
  * The readout's measured fields are all functions of the count DISTRIBUTION: usage_frac,
    used_experts, entropy_norm, and load_gini (normalised by tot). Only `tokens` (2x) and
    `window_steps` move.
So that refusal's specific claim -- that usage fraction, entropy and Gini would be "computed
over a doubled denominator, reporting a healthier load spread" -- is FALSE for a uniform double
count. This script measures whether the doubling is in fact uniform, which is what makes that
statement checkable rather than merely doubted: a NON-uniform double count would move the
statistics, and is the case that would justify the guard as written.

REFUSES rather than reporting a number it cannot stand behind:
  * no CUDA -> refuse (a CPU run cannot execute this model)
  * either arm counting zero tokens -> refuse (an all-zero counter compares equal to another
    all-zero counter, so "equal" would be indistinguishable from "nothing ran")
  * the two arms disagreeing on which layers are MoE, or on the routed-expert count -> refuse
  * grad_ckpt not actually active in the ON arm -> refuse, because the whole test is that one
    line; `ckpt = self.grad_ckpt and self.training` needs BOTH, so an arm left
    in eval mode would silently measure the OFF condition twice and print "equal".
"""
import argparse
import json
import os
import sys

# restartable: two forward+backward passes on a 4-block model with a fixed seed, ~seconds on one
# card, and it writes nothing until both arms have finished. An interrupt loses the run and
# nothing else -- re-running reproduces the same counts exactly, because the input is
# torch.manual_seed(seed) and randint, not sampled data. No checkpoint, no shard, no ledger row.

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))


SHAPE = dict(d=256, heads=2, layers=4, ffn_hidden=256, vocab=256, seq=64, attn_every=2,
             moe_experts=8, moe_layers="0-3", moe_top_k=3, moe_shared=1, moe_expert_ffn=64)


def check_shape(overrides=None):
    """The two constraints this script must enforce itself, checked WITHOUT a card.

    Both live in code that this script bypasses. FlashKDA pins head_dim to 128 (HeadMix's raise,
    `head_dim is pinned to ... by the FlashKDA CUTLASS kernel`) and train.py's main() raises
    `head_dim must be 128`, but this script builds HybridLM directly so that raise never runs.
    MoEFFN enforces (moe_top_k + moe_shared) * moe_expert_ffn == ffn_hidden so a loss delta is
    attributable to sparsity rather than FLOPs, and refuses in its constructor otherwise -- which
    costs a card claim to discover.

    NEITHER IS THE CAUSE OF THE `misaligned address` FAULT. I blamed head_dim for it, changed the
    shape, spent a second claim, and got the identical fault at head_dim 128. That fault is fp32;
    see the bf16 note in build(). These checks are here to save a claim, not because they fix it.
    """
    s = dict(SHAPE)
    # A TYPO'D FIELD IS REFUSED HERE, not in build(): build imports torch, so a misspelled
    # override would only be caught on a card. SHAPE is the authority for what this script sets;
    # anything outside it is either a typo or a field this script has no business overriding.
    unknown = sorted(k for k in (overrides or {}) if k not in SHAPE)
    if unknown:
        sys.exit(f"REFUSING: --set names {', '.join(unknown)!r}, not in this script's shape "
                 f"({', '.join(sorted(SHAPE))}). A typo would otherwise set an attribute the "
                 f"model never reads, and the run would look like it honoured the override")
    s.update(overrides or {})
    if s["d"] % s["heads"] or s["d"] // s["heads"] != 128:
        sys.exit(f"REFUSING: d {s['d']} / heads {s['heads']} = {s['d'] / s['heads']:g}, but "
                 f"FlashKDA pins head_dim to 128. train.py's main() raises on this; building "
                 f"HybridLM directly does not. (This is NOT the cause of the `misaligned "
                 f"address` fault -- that one is fp32, and it persisted unchanged at head_dim "
                 f"128.) Set d = heads * 128")
    active = (s["moe_top_k"] + s["moe_shared"]) * s["moe_expert_ffn"]
    if active != s["ffn_hidden"]:
        sys.exit(f"REFUSING: (moe_top_k {s['moe_top_k']} + moe_shared {s['moe_shared']}) * "
                 f"moe_expert_ffn {s['moe_expert_ffn']} = {active} != ffn_hidden "
                 f"{s['ffn_hidden']}; MoEFFN enforces equal-active parity and would refuse in its "
                 f"constructor. Checked here so an override fails before a card is touched")
    return s


def build(grad_ckpt, seed=42, shape=None):
    """A small MoE model on the card. grad_ckpt is set AFTER __init__ deliberately.

    HybridLM.__init__ raises for moe_experts + grad_ckpt in the constructor, and that guard is the
    thing under test -- it may only be deleted once this measurement says the counter is safe.
    Setting the attribute afterwards reaches the same code path the guard protects
    (`ckpt = self.grad_ckpt and self.training` in HybridLM._body) WITHOUT deleting the guard first,
    so this script never needs an unguarded tree to run in. It also means the flag I set is the
    flag the forward reads, which is asserted below rather than assumed.

    THE CONFIG IS TRAIN.PY'S OWN Cfg, mutated, not a hand-rolled namespace. My first version
    invented a class with cfg.dim and called HybridLM `Transformer` -- both wrong (the field is
    cfg.d, the class is HybridLM), and either would have died on the card after the grant was
    already spent. Taking the real Cfg means every field HybridLM reads exists with the
    launcher's own default, and only the shape is shrunk.
    """
    import torch

    from train import Cfg, HybridLM

    # attn_every stays >= 2: HybridLM REFUSES attn_every=1 ("0 KDA layers, but GatedMLA is
    # NoPE"), so a KDA layer -- hence a card -- is unavoidable. 4 blocks, 2 of them MoE.
    #
    # HEAD_DIM MUST BE 128, so d = heads * 128. The FlashKDA CUTLASS kernel pins it and
    # train.py's main() raises `head_dim must be 128`, which this script must enforce itself
    # because it constructs HybridLM DIRECTLY and that raise lives in main().
    #
    # THIS WAS NOT THE CAUSE OF THE `misaligned address` FAULT, and I asserted that it was.
    # I changed d=128,heads=4 -> d=256,heads=2 on that theory, spent a second card claim, and got
    # the identical fault at head_dim 128. The cause is dtype (see the bf16 note below). The
    # head_dim constraint is real and worth enforcing; it was simply not this bug.
    Cfg.d, Cfg.heads, Cfg.layers, Cfg.ffn_hidden = 256, 2, 4, 256
    Cfg.vocab = Cfg.vocab_real = 256
    Cfg.seq, Cfg.fone = 64, False
    Cfg.attn_every = 2
    # EQUAL-ACTIVE PARITY IS ENFORCED BY MoEFFN, not optional:
    # (moe_top_k + moe_shared) * moe_expert_ffn must equal ffn_hidden EXACTLY or it refuses,
    # so that a loss delta is attributable to sparsity rather than FLOPs. My first version set
    # top_k and left moe_expert_ffn at train.py's default 768, and the card rejected it in the
    # constructor: "active FFN width 2304 != dense ffn_hidden 256". With ffn_hidden 256 and
    # moe_shared 1, top_k 3 needs moe_expert_ffn 64 -- (3+1)*64 = 256.
    Cfg.moe_experts, Cfg.moe_layers, Cfg.moe_top_k = 8, "0-3", 3
    Cfg.moe_shared, Cfg.moe_expert_ffn = 1, 64
    Cfg.grad_ckpt = False          # constructed OFF; see docstring
    # EVERY SHAPE FIELD IS OVERRIDABLE FROM THE CLI, because both of this script's failures so
    # far were config constraints that exist only on the card (MoEFFN's parity raise, then
    # FlashKDA's head_dim) and each one cost a commit, a merge and a re-claim to change one
    # integer. `--set d=512,heads=4` does not re-derive the constraints above -- they still
    # apply -- but it makes the next one a flag instead of a merge cycle.
    for k, v in (shape or {}).items():
        setattr(Cfg, k, v)   # names already validated against SHAPE by check_shape()
    torch.manual_seed(seed)
    # BF16, AND THIS IS THE FIX FOR THE `misaligned address` FAULT, not a performance choice.
    # HybridLM(Cfg) builds fp32 parameters and the KDA Triton kernel faults in fp32 with
    # `CUDA error: misaligned address` inside its autotuner. eval_heldout.py's loader carries the
    # diagnosis and the exclusions someone already paid for: row width (64 fails as readily as
    # 4096), cu=None vs doc_cu_seqlens, FLA_FLASH_KDA=0 before any import, grad checkpointing --
    # all excluded, dtype was the one thing that differed from every path that works. I burned two
    # card claims blaming head_dim before reading that comment.
    m = HybridLM(Cfg).cuda().to(torch.bfloat16).train()
    m.grad_ckpt = grad_ckpt
    return m, Cfg


def moe_layers_of(m):
    """(index, module) for every block whose ffn carries tokens_per_expert."""
    out = []
    for i, b in enumerate(m.blocks):
        for name, sub in b.named_modules():
            if hasattr(sub, "tokens_per_expert"):
                out.append((i, name, sub))
    return out


def run_arm(grad_ckpt, seed, steps, shape=None, skip_commit=False):
    import torch
    m, cfg = build(grad_ckpt, seed, shape)
    layers = moe_layers_of(m)
    if not layers:
        sys.exit("REFUSING: no module with tokens_per_expert was found; the model shape changed "
                 "and this test would compare nothing")
    for _i, _n, sub in layers:
        sub.tokens_per_expert.zero_()
        sub.windows.zero_()

    # THE FLAG MUST BE LIVE, not merely set. `ckpt = self.grad_ckpt and self.training` needs
    # both, so an arm in eval() would measure the OFF condition twice and print "equal".
    if bool(getattr(m, "grad_ckpt", False)) != bool(grad_ckpt):
        sys.exit(f"REFUSING: m.grad_ckpt is {getattr(m, 'grad_ckpt', None)!r}, asked for "
                 f"{grad_ckpt!r} -- the attribute name changed and this arm is not the arm")
    if not m.training:
        sys.exit("REFUSING: model is not in train() mode, so grad_ckpt is inert "
                 "(`ckpt = self.grad_ckpt and self.training` requires it) and both arms would measure OFF")

    torch.manual_seed(seed)
    ids = torch.randint(0, cfg.vocab, (2, cfg.seq), device="cuda")
    for _ in range(steps):
        # `m(x, x)` and a 2-tuple return, both taken from scripts/probe_gradckpt_sources.py
        # rather than guessed -- that probe is the one script in the tree that already runs
        # this model with grad_ckpt off and on.
        h, _ = m(ids, ids)
        h.float().mean().backward()   # backward is what triggers recompute
        if not skip_commit:
            for _i, _n, sub in layers:
                if hasattr(sub, "commit_token_counts"):
                    sub.commit_token_counts()
        m.zero_grad(set_to_none=True)

    # COMMIT ONCE PER MICRO-BATCH, exactly where train.py does it: after backward, in the loop
    # body. Without this the ON arm keeps the recompute surplus and the test measures the bug
    # instead of the fix -- so this call is what makes the post-fix run meaningful, and its
    # absence is what the pre-fix run measured.
    counts = {f"{i}.{n}": sub.tokens_per_expert.detach().cpu().tolist()
              for i, n, sub in layers}
    windows = {f"{i}.{n}": int(sub.windows) for i, n, sub in layers}
    return counts, windows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--json", help="write the result here")
    ap.add_argument("--no-commit", action="store_true",
                    help="skip commit_token_counts() -- reproduces the PRE-FIX behaviour, which "
                         "is how the 2.0000 ratio was originally measured")
    ap.add_argument("--set", dest="overrides", default="",
                    help="comma-separated Cfg overrides, e.g. d=512,heads=4. Ints are parsed as "
                         "ints; everything else stays a string (moe_layers is '0-3')")
    a = ap.parse_args()

    # OVERRIDES ARE PARSED AND SHAPE-CHECKED BEFORE THE CUDA CHECK, deliberately: a typo or an
    # illegal shape must be catchable on a machine with no card. Otherwise the only way to find
    # out that `--set d=128,heads=4` is invalid is to claim a card, push, and watch the kernel
    # fault -- which is exactly the loop that cost this script two grants.
    shape = {}
    for item in filter(None, (t.strip() for t in a.overrides.split(","))):
        if "=" not in item:
            sys.exit(f"REFUSING: --set item {item!r} has no '='")
        k, v = item.split("=", 1)
        shape[k.strip()] = int(v) if v.strip().lstrip("-").isdigit() else v.strip()
    if shape:
        print(f"overrides: {shape}")
    check_shape(shape)

    import torch
    if not torch.cuda.is_available():
        sys.exit("REFUSING: no CUDA. Every valid config carries a KDA layer (HybridLM refuses "
                 "attn_every=1: '0 KDA layers, but GatedMLA is NoPE'), KDA needs chunk_kda, and "
                 "chunk_kda is None without the Triton build. Stubbing it -- as "
                 "probe_gradckpt_sources.py does -- would answer this question on a different "
                 "model, and 'reading a plausible mechanism is not observing it' is the whole "
                 "reason this test exists. Run on the granted card.")
    print(f"card: {torch.cuda.get_device_name(0)}  "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")

    off_c, off_w = run_arm(False, a.seed, a.steps, shape, a.no_commit)
    on_c, on_w = run_arm(True, a.seed, a.steps, shape, a.no_commit)

    if set(off_c) != set(on_c):
        sys.exit(f"REFUSING: the two arms disagree on which layers are MoE: "
                 f"off={sorted(off_c)} on={sorted(on_c)}")
    tot_off = sum(sum(v) for v in off_c.values())
    tot_on = sum(sum(v) for v in on_c.values())
    if tot_off == 0 or tot_on == 0:
        sys.exit(f"REFUSING: an arm counted zero tokens (off={tot_off}, on={tot_on}). Two "
                 f"all-zero counters compare EQUAL, so this would print 'equal' for a run that "
                 f"never happened")

    print(f"\n{'layer':16}{'sum off':>10}{'sum on':>10}{'ratio':>8}  {'win off':>8}{'win on':>8}"
          f"  uniform?")
    ratios, nonuniform = [], []
    for k in sorted(off_c, key=lambda s: (int(s.split('.')[0]), s)):
        a_, b_ = off_c[k], on_c[k]
        sa, sb = sum(a_), sum(b_)
        r = sb / sa if sa else float("nan")
        ratios.append(r)
        # Per-expert uniformity: the statistics only survive a doubling if EVERY expert doubles.
        per = {round(y / x, 4) for x, y in zip(a_, b_) if x}
        uni = len(per) == 1
        if not uni:
            nonuniform.append((k, sorted(per)[:5]))
        print(f"{k:16}{sa:10d}{sb:10d}{r:8.4f}  {off_w[k]:8d}{on_w[k]:8d}  "
              f"{'yes' if uni else 'NO ' + str(sorted(per)[:4])}")

    lo, hi = min(ratios), max(ratios)
    print(f"\ntotal off {tot_off}  on {tot_on}  ratio {tot_on / tot_off:.4f}  "
          f"(per-layer {lo:.4f}-{hi:.4f})")
    verdict = ("EQUAL: recompute does NOT double the counter; the guard refuses a configuration "
               "that was never broken in this respect"
               if abs(hi - 1.0) < 1e-9 and abs(lo - 1.0) < 1e-9 else
               "DOUBLED: ratio 2 on the MoE layers, the signature the task predicted"
               if abs(lo - 2.0) < 1e-9 and abs(hi - 2.0) < 1e-9 else
               f"NEITHER 1 NOR 2: ratios span {lo:.4f}-{hi:.4f}, so the defect is not a clean "
               f"double and the fix cannot be a division")
    print(f"verdict: {verdict}")
    if nonuniform:
        print(f"NON-UNIFORM on {len(nonuniform)} layer(s) -- this is the case that WOULD move "
              f"usage_frac/entropy/gini, since those are scale-invariant only under a uniform "
              f"factor: {nonuniform[:3]}")
    else:
        print("uniform across experts on every layer, so entropy_norm/load_gini/usage_frac are "
              "unaffected either way and only `tokens`/`window_steps` move")

    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump({"off": off_c, "on": on_c, "windows_off": off_w, "windows_on": on_w,
                       "ratio_total": tot_on / tot_off, "ratio_min": lo, "ratio_max": hi,
                       "nonuniform_layers": [k for k, _ in nonuniform],
                       "steps": a.steps, "seed": a.seed}, fh, indent=2)
        print(f"wrote {a.json}")


if __name__ == "__main__":
    main()
