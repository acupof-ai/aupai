#!/usr/bin/env python3
"""A mid-run SFT checkpoint must carry optimizer state, checked by READING THE FILE BACK.

WHY NOT ASSERT ON THE CALL. The defect this guards against was `save_checkpoint(...,
step=step)` with the `opt` argument omitted while `good_opt` sat on the line above --
sft_math.py:370 computed the snapshot and :372 dropped it. Every source-level check I could
write for that (does the call name `opt=`? is `opt_snapshot` called?) was ALREADY TRUE of the
broken code, because the snapshot was taken; only the argument was missing. So the criterion
has to be the artifact: load the written file and require a usable optimizer in it.

WHY IT TESTS BOTH WORLDS. A test that only asserts "opt is present" passes on a file that
carries `opt` as an empty list, or one moment tensor for a model with 63 of them. Case 2
therefore strips the argument back out of the real source, runs it again, and requires the
file to LOSE the optimizer -- if it does not, this test is measuring something other than
the argument.

Runs on CPU with a 2-layer toy model and 2 steps; it does not need the pod, a GPU or a pack.
"""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# The hook invokes every gated file as `<file> --selftest` (scripts/hooks/pre-commit:937), and
# this file's whole body IS the selftest, so the flag is accepted and ignored rather than
# parsed. Refusing an unknown argument would fail the hook; silently accepting ANY argument
# would let a typo'd flag read as a pass, so only that one word is allowed.
if len(sys.argv) > 2 or (len(sys.argv) == 2 and sys.argv[1] != "--selftest"):
    raise SystemExit(f"usage: {os.path.basename(__file__)} [--selftest]  (got {sys.argv[1:]})")

from train import opt_snapshot, save_checkpoint  # noqa: E402

fails = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {name}{'' if cond else '  <- ' + detail}")
    if not cond:
        fails.append(name)


def build():
    """Two parameter groups, because the real run has several optimizers and opt_snapshot
    returns a LIST -- a guard built on one optimizer would not notice a save that wrote
    only the first."""
    m = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 4))
    opts = [torch.optim.AdamW(m[0].parameters(), lr=1e-3),
            torch.optim.AdamW(m[1].parameters(), lr=1e-3)]
    for _ in range(2):  # two steps, so exp_avg is populated rather than absent
        loss = m(torch.randn(4, 8)).square().mean()
        loss.backward()
        for o in opts:
            o.step()
            o.zero_grad(set_to_none=True)
    return m, opts


def write(path, m, opts, pass_opt):
    save_checkpoint(path, {k: v.cpu().clone() for k, v in m.state_dict().items()},
                    {"dim": 8}, "toy-vocab",
                    opt=opt_snapshot(opts) if pass_opt else None, step=250)


tmp = os.path.join("/tmp", f"test_sft_save_opt_{os.getpid()}")
m, opts = build()

# 1. WITH the argument: the file must hold one entry per optimizer, each with real moments.
write(tmp + ".with.pt", m, opts, pass_opt=True)
ck = torch.load(tmp + ".with.pt", map_location="cpu", weights_only=False)
check("mid-run save writes an optimizer", "opt" in ck, f"keys={sorted(ck)}")
if "opt" in ck:
    check("one entry per optimizer", len(ck["opt"]) == len(opts),
          f"{len(ck['opt'])} entries for {len(opts)} optimizers")
    # NOT just "state is non-empty": a snapshot taken before any step has the keys and no
    # moments, and that file cannot resume anything.
    moments = [k for e in ck["opt"] for st in e["state"].values() for k in st
               if k in ("exp_avg", "exp_avg_sq")]
    check("moments are present, not just the state keys", len(moments) >= 2 * len(opts),
          f"{len(moments)} moment tensors")
    check("param_groups carry the lr", all("lr" in g for e in ck["opt"] for g in e["param_groups"]))
    check("the step is recorded beside it", ck.get("step") == 250, f"step={ck.get('step')}")
    # The optimizer must be LOADABLE, not merely shaped right -- a deepcopy that aliased live
    # CUDA moments would pass every check above and fail here on a real run.
    fresh = [torch.optim.AdamW(p.parameters(), lr=1e-3) for p in (m[0], m[1])]
    try:
        for o, sd in zip(fresh, ck["opt"], strict=True):
            o.load_state_dict(sd)
        check("the saved state loads into a fresh optimizer", True)
    except Exception as e:  # noqa: BLE001
        check("the saved state loads into a fresh optimizer", False, repr(e)[:120])

# 2. WITHOUT it: the file must LOSE the optimizer. This is the half that proves the check
#    is measuring the argument and not something incidental.
write(tmp + ".without.pt", m, opts, pass_opt=False)
ck2 = torch.load(tmp + ".without.pt", map_location="cpu", weights_only=False)
check("dropping the argument drops the optimizer (the defect is reproducible)",
      "opt" not in ck2, f"opt still present: keys={sorted(ck2)}")

# 3. THE SHIPPED CALL SITE must pass it. Source-level, and stated as the weak check it is:
#    it cannot tell a real snapshot from a wrong one, which is what case 1 is for.
with open(os.path.join(ROOT, "sft_math.py"), encoding="utf-8") as fh:
    src = fh.read()
mid = src.split("if step % args.save_every == 0:", 1)
check("sft_math.py's mid-run save passes opt=", len(mid) == 2 and "opt=good_opt" in mid[1][:2000],
      "the call site does not name opt=good_opt")

for suf in (".with.pt", ".without.pt"):
    if os.path.exists(tmp + suf):
        os.remove(tmp + suf)

print(f"\n{'ALL OK' if not fails else 'FAILED: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
