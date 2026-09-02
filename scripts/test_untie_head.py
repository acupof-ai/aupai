#!/usr/bin/env python3
# restartable: builds tiny CPU models, no data, no card. Seconds.
"""b0-17's three arms must be three DIFFERENT experiments, and two of them look identical in logs.

    python3 scripts/test_untie_head.py --selftest

THE ARMS:
  1  tied                 head.weight IS tok.weight, trained at embed_lr 0.1 (today's behaviour)
  2  --untie_head         own head weights, still in the embed group at 0.1
  3  --untie_head --head_lr 0.003464   own weights AND its own AdamW group

Arm 2 exists to isolate CAPACITY (+33,619,968 params, +16.3%, the same magnitude as A/B (4)'s
value_embed table) from the lr MECHANISM, which is what arm 2 -> arm 3 isolates. That only works
if the arms really differ, and the ways they can silently collapse are not hypothetical:

  - Arms 2 and 3 have IDENTICAL parameter counts and identical logs apart from one lr field. If
    the grouping branch in build_optimizers misfires, arm 3 becomes arm 2 and two different
    experiments print the same number. Nothing errors.
  - "Untied" that forgets to break the alias adds no parameters -- but a version that adds the
    tensor and then re-aliases it would be invisible to a parameter count, which can only show a
    table exists, not that it is unshared. So `is not` is asserted directly.
  - The vocab pad rows (vocab_real:vocab) are zeroed through head.weight at model.py:447. While
    tied, one zero_() does both tensors; untied, tok's pad rows keep their std=0.02 draw and the
    untied arm trains pad rows the tied arm never touched. That is a hidden variable in an A/B
    about the head (1e caught this before the run).
  - `_fp8_ok` excludes the head BY NAME (train.py:362), and untying must not quietly send a
    33.6M-parameter matmul into fp8 -- the audit table records head as fp8-excluded.

Every check below was verified red on its own broken world.
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("FLA_FLASH_KDA", "0")

HEAD_LR_D1024 = 0.003464  # nanochat 0.004*(d/768)**-0.5 at d=1024


def build(untie, head_lr, d=128, layers=4):
    """A tiny model plus its optimizers, with the arm's flags applied to Cfg.

    Small shapes on purpose: the trap is in the wiring, and a d1024 model costs 30s per arm to
    prove the same thing. The parameter-count arithmetic is checked separately against the real
    width, where the +16.3% claim actually lives.
    """
    import torch

    import model as M
    import train
    M.chunk_kda = lambda q, k, v, **kw: (q * 0 + v, None)  # noqa: E731  no kernel needed
    M.HAS_FA = False
    c = train.Cfg
    c.d, c.layers, c.seq, c.batch, c.accum = d, layers, 64, 1, 1
    c.attn_every, c.attn_res = 2, True
    c.untie_head, c.head_lr = untie, head_lr
    torch.manual_seed(42)
    m = M.HybridLM(c)
    return m, train.build_optimizers(m, c), train, c


def head_group_lr(m, opts):
    """The lr the HEAD parameter actually receives -- found by identity, not by group name.

    Reading the group called "head" would pass on a build where the head never reached that group,
    because the group would simply be absent and the lookup would return None. Searching for the
    tensor answers the question that matters: what lr does this weight get?
    """
    for o in opts:
        for g in o.param_groups:
            if any(p is m.head.weight for p in g["params"]):
                return g["lr"]
    return None


def group_lr(opts, name):
    for o in opts:
        if getattr(o, "aupai_group", None) == name:
            return o.param_groups[0]["lr"]
    return None


def _selftest():
    import train
    fails = []

    # ARM 1: tied. The alias holds and the head trains at the embedding lr -- this is the
    # behaviour every existing checkpoint has, so a change here is a silent recipe change.
    m1, o1, _, _ = build(False, 0.0)
    if m1.head.weight is not m1.tok.weight:
        fails.append("arm 1 (default) is UNTIED: the default changed, so every run that does not "
                     "pass --untie_head silently switched architecture")
    if head_group_lr(m1, o1) != group_lr(o1, "embed"):
        fails.append(f"arm 1's head lr {head_group_lr(m1, o1)} != embed lr {group_lr(o1, 'embed')}")
    if group_lr(o1, "head") is not None:
        fails.append("arm 1 built a separate head optimizer group; the tied head must ride embed")

    # ARM 2: untied, no head_lr. Own weights, still at embed lr. This is the capacity arm.
    m2, o2, _, _ = build(True, 0.0)
    if m2.head.weight is m2.tok.weight:
        fails.append("arm 2 asked for --untie_head and the head is STILL aliased to tok, so the "
                     "arm adds no parameters and measures nothing -- and a parameter count cannot "
                     "see this, since a re-aliased tensor still exists")
    if head_group_lr(m2, o2) != group_lr(o2, "embed"):
        fails.append(f"arm 2's head lr {head_group_lr(m2, o2)} != embed lr "
                     f"{group_lr(o2, 'embed')}: arm 2 IS arm 3 and capacity is no longer isolated")

    # ARM 3: untied with its own lr. THE ARM THAT CAN COLLAPSE INTO ARM 2 WITHOUT ERRORING.
    m3, o3, _, _ = build(True, HEAD_LR_D1024)
    if m3.head.weight is m3.tok.weight:
        fails.append("arm 3's head is aliased to tok")
    got = head_group_lr(m3, o3)
    if got != HEAD_LR_D1024:
        fails.append(f"arm 3's head trains at {got}, not {HEAD_LR_D1024}. If it equals the embed "
                     f"lr, arm 3 has silently become arm 2 -- identical parameter count, "
                     f"identical logs, two experiments reporting one number")
    if group_lr(o3, "head") != HEAD_LR_D1024:
        fails.append("arm 3 has no optimizer group named 'head', so the step line cannot show its "
                     "lr and b0-14's per-group logging is blind to this arm")

    # ARMS 2 AND 3 MUST MATCH ON PARAMETERS AND DIFFER ONLY IN LR. That is the whole design: if
    # their counts diverge, the arm-2 -> arm-3 comparison stops isolating the lr.
    n2 = sum(p.numel() for p in m2.parameters())
    n3 = sum(p.numel() for p in m3.parameters())
    if n2 != n3:
        fails.append(f"arms 2 and 3 differ in parameter count ({n2} vs {n3}); the lr comparison is "
                     f"confounded by capacity, which is the confound arm 2 exists to remove")
    n1 = sum(p.numel() for p in m1.parameters())
    if n2 <= n1:
        fails.append(f"untying did not ADD parameters ({n1} -> {n2}); the head is not a real "
                     f"second tensor")

    # THE PAD ROWS, BOTH TENSORS. Untied, tok's alignment padding must still be zero: while tied,
    # model.py's single zero_() covered both, and untying it would leave the untied arm training
    # pad rows the tied arm never touched.
    real = getattr(train.Cfg, "vocab_real", train.Cfg.vocab)
    if real < train.Cfg.vocab:
        for tag, mm in (("tied", m1), ("untied", m2)):
            for which, t in (("head", mm.head.weight), ("tok", mm.tok.weight)):
                pad = t[real:train.Cfg.vocab]
                if pad.numel() and pad.abs().max().item() != 0.0:
                    fails.append(f"{tag} arm's {which} alignment padding is nonzero "
                                 f"(max |w| {pad.abs().max().item():.3e}); random pad logits steal "
                                 f"softmax denominator mass (eff.vocab_padding_softmax_defect) and "
                                 f"in the untied arm they also become trainable, which the tied "
                                 f"arm's pad rows are not")
    else:
        fails.append(f"vocab_real {real} is not below vocab {train.Cfg.vocab}, so the pad-row "
                     f"check tested nothing -- if the tokenizer changed, re-derive this check "
                     f"rather than deleting it")

    # FP8: the head stays excluded BY NAME, untied or not. Untying makes it a standalone
    # 33.6M-parameter matmul, and the audit table records the head as fp8-excluded.
    for tag, mm in (("tied", m1), ("untied", m2)):
        if train._fp8_ok(mm.head, "head"):
            fails.append(f"{tag} arm's head passes _fp8_ok, so untying sent a 33.6M matmul into "
                         f"fp8 that the audit table records as excluded")

    # THE REAL WIDTH's arithmetic, which the tiny models cannot show: the untied head is exactly
    # one padded_vocab x d tensor, and at d1024 that is the +16.3% that makes this arm share A/B
    # (4)'s capacity confound.
    pv, d = 32832, 1024
    if pv * d != 33619968:
        fails.append(f"padded_vocab*d is {pv * d}, not the 33,619,968 quoted in the exp row")
    base = 206128200
    pct = pv * d / base * 100
    if not (16.0 < pct < 16.6):
        fails.append(f"the untied head is +{pct:.1f}% of the 206.1M baseline, not the +16.3% the "
                     f"task and the A/B (4) comparison both quote")

    if fails:
        for f in fails:
            print(f"FAIL: {f}", file=sys.stderr)
        return 1
    print("test_untie_head selftest OK: all three arms are distinct -- tied keeps the alias and "
          "rides embed lr, untied@embed adds its own tensor at the SAME lr (isolating capacity), "
          "untied@own gets its own optimizer group at 0.003464 (isolating the lr). Arms 2 and 3 "
          "match on parameter count and differ only in head lr, which is the design: arm 3 "
          "collapsing into arm 2 is the failure that errors nowhere and prints one number for two "
          "experiments. The head lr is read by TENSOR IDENTITY, not by group name, so an absent "
          "group cannot pass as a match. Alignment pad rows are zero in BOTH head and tok for the "
          "untied arm (tied, one zero_() covered both; untied, tok's pad rows would otherwise "
          "train in only one arm). The head stays fp8-excluded by name in both arms. And the real "
          "width's arithmetic is pinned: 32832*1024 = 33,619,968 = +16.3% of 206.1M, the same "
          "capacity confound A/B (4) carried.")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.parse_args()
    return _selftest()


if __name__ == "__main__":
    sys.exit(main())
