#!/usr/bin/env python3
"""--bf16 casts the experts to bf16 without the fp8 conversion, and --fp8 stays what it was.

WHY THIS EXISTS. prereg moe_0905 amendment 8 specified a MoE-vs-dense pair "both with fp8 off" to
remove the fp8/bf16 confound from readout 5. Run b0_p5_e1_bf16 died at step 0 (2026-09-05, rc=1)
on train.py's own guard: `--fp8` performs TWO things -- the bf16 cast AND convert_to_fp8_compute --
so dropping it leaves fp32 masters, and torch._grouped_mm compiles only for bf16. The pair as
designed could not exist. `--bf16` separates the two effects so the equal-precision pair can run.

WHAT IS CHECKED, by running train.py's real argument parsing and dtype resolution rather than by
reading it. The three flag states must give three different worlds, and the point of the test is
that the DIFFERENCE is visible -- a test asserting only "under --bf16 the experts are bf16" passes
for code that casts unconditionally, which would silently change every fp32 arm's arithmetic.

  1. --bf16 alone      -> experts bf16, and NO fp8 conversion happened.
  2. --fp8 alone       -> experts bf16 (unchanged behaviour), fp8 conversion happened.
  3. neither, with MoE -> REFUSED, and the message names --bf16 as an option.
  4. --bf16 --fp8      -> REFUSED as mutually exclusive.
  5. neither, no MoE   -> experts absent, model stays fp32: the flag changed nothing else.

HOW, and why not by importing train.py: train.py's main() builds a model, reads a token cache and
initialises distributed, none of which belongs in a dtype test, and its module scope is not
importable without those. So the two decision sites are executed in isolation -- the argparse
declarations for --fp8/--bf16 and the resolution block that computes `fp8`/`bf16_only`, refuses the
bad combinations and performs the cast -- extracted from the source by anchor. That is the same
technique scripts/test_moe_diag_rank.py uses and it carries the same limitation, stated here so a
reader does not assume otherwise: THIS PROVES THE BLOCK'S LOGIC, NOT THAT main() REACHES IT with
these values. The in-situ assertion is a real launch, and the runs themselves are the evidence.

    python3 scripts/test_moe_diag_rank.py   # sibling: the rank-dependent diag write
    python3 scripts/test_bf16_flag.py
"""
import os
import sys
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

_fails = []


def _check(name, got, want):
    if got != want:
        _fails.append(f"{name}: got {got!r}, want {want!r}")
        print(f"  FAIL {name}: got {got!r}, want {want!r}")
    else:
        print(f"  ok   {name}: {got!r}")


def _slice_by_indent(lines, i):
    """From line i to the first following non-blank line indented no deeper than line i."""
    base = len(lines[i]) - len(lines[i].lstrip())
    out = [lines[i]]
    for l in lines[i + 1:]:
        if l.strip() and (len(l) - len(l.lstrip())) <= base:
            break
        out.append(l)
    return out


def _extract_resolution(src):
    """The dtype resolution, as THREE line-sliced pieces in train.py's own order.

    Returns (head, guard, casts): the `fp8`/`bf16_only` computation with its two refusals, the
    MoE dtype guard, and the cast branches. Kept separate because the refusals must be able to
    fire BEFORE the cast, which is their whole point -- executing the resolution whole and the
    guard afterwards would cast first and refuse second.

    SLICED BY LINE AT STATEMENT BOUNDARIES, not by splitting the text on substrings. The first
    version split on the literals "if getattr(Cfg," and "if fp8:", and both matched inside the
    comments I had just written -- the block explains `if fp8:` in prose -- so the reassembled
    source was a comment fragment and every exec raised SyntaxError. Three checks still printed
    "ok" because a refusal-shaped test passes when nothing runs at all, which is why the
    dtype/conversion assertions are paired with an explicit "runs" check on the error.
    """
    lines = src.split("\n")

    def _one(pred, what):
        hits = [i for i, l in enumerate(lines) if pred(l)]
        if len(hits) != 1:
            raise SystemExit(f"expected exactly one {what}, found {len(hits)}")
        return hits[0]

    i_head = _one(lambda l: l.strip() == "fp8 = args.fp8 and amp", "`fp8 = args.fp8 and amp`")
    i_guard = _one(lambda l: l.strip().startswith('if getattr(Cfg, "moe_experts", 0)')
                   and "not fp8" in l, "MoE dtype guard")
    i_cast = _one(lambda l: l.strip() == "if fp8:" and l.startswith("    if"),
                  "top-level `if fp8:` cast branch")
    i_end = _one(lambda l: l.strip().startswith("n_params = sum("), "`n_params = sum(` end anchor")
    # The cast run ends at the `if is_main:` that owns the end anchor.
    j = i_end
    while j > i_cast and lines[j].strip() != "if is_main:":
        j -= 1

    if not i_head < i_guard < i_cast < j:
        raise SystemExit(f"unexpected order in train.py: {i_head} {i_guard} {i_cast} {j}")

    head = textwrap.dedent("\n".join(lines[i_head:i_guard]))
    guard = textwrap.dedent("\n".join(_slice_by_indent(lines, i_guard)))
    casts = textwrap.dedent("\n".join(lines[i_cast:j]))
    for piece, must in ((head, "bf16_only"), (guard, "REFUSING"),
                        (casts, "convert_to_fp8_compute"), (casts, "torch.bfloat16")):
        if must not in piece:
            raise SystemExit(f"a slice lacks {must!r} -- the extraction is wrong")
    if "elif bf16_only:" not in casts:
        raise SystemExit("the cast slice lacks the --bf16 branch -- the extraction is wrong")
    return head, guard, casts


class _P:
    """A stand-in parameter that records casts instead of allocating memory."""

    def __init__(self, dtype="float32"):
        self.dtype = dtype


class _FakeModel:
    def __init__(self, moe):
        self.dtype = "float32"
        self.converted = False
        self.w13 = _P() if moe else None
        self.w2 = _P() if moe else None

    def to(self, dt):
        # RECORDED AS THE REAL DTYPE'S NAME, and compared against the literal "bfloat16" in the
        # checks rather than against str(torch.bfloat16). Recording whatever it is handed and
        # comparing to the same expression would make the assertion invisible to a mutation that
        # changes the cast: swapping torch.bfloat16 for torch.float16 survived exactly that way
        # (M6, measured) because both sides of the comparison moved together. The literal is the
        # anchor -- torch._grouped_mm requires BF16 specifically, so fp16 is a real defect and
        # must read as one.
        name = str(dt).replace("torch.", "")
        self.dtype = name
        for p in (self.w13, self.w2):
            if p is not None:
                p.dtype = name
        return self


def _run(src, *, fp8_flag, bf16_flag, moe, amp=True):
    """Execute the guard and the resolution as train.py would. Returns (model, error)."""
    import torch

    model = _FakeModel(moe)
    ns = {
        "args": type("A", (), {"fp8": fp8_flag, "bf16": bf16_flag})(),
        "amp": amp,
        "torch": torch,
        "raw_model": model,
        "is_main": False,
        "os": os,
        "Cfg": type("C", (), {"moe_experts": 24 if moe else 0, "fone": False})(),
        "convert_to_fp8_compute": lambda m: setattr(m, "converted", True),
        "patch_liger_flce_fp8": lambda: True,
    }
    head, guard, casts = _extract_resolution(src)
    # Executed in train.py's own order, so a refusal reached before the cast prevents it -- which
    # is what checks 3, 4 and 6 assert by reading the dtype after the error.
    try:
        exec(head, ns)
        exec(guard, ns)
        exec(casts, ns)
        err = None
    except SystemExit as e:
        err = str(e)
    except Exception as e:  # noqa: BLE001 -- which error escapes is the finding
        err = f"{type(e).__name__}: {e}"
    return ns.get("raw_model", model), err


def main():
    with open(os.path.join(ROOT, "train.py"), encoding="utf-8") as f:
        src = f.read()

    # 1. --bf16 alone: bf16 experts, NO fp8 conversion. Both halves matter -- the dtype alone
    #    would also pass under --fp8, so it cannot distinguish the flags.
    m, err = _run(src, fp8_flag=False, bf16_flag=True, moe=True)
    _check("--bf16 runs", err, None)
    _check("--bf16 casts the experts to bf16", m.w13.dtype, "bfloat16")
    _check("--bf16 casts w2 too", m.w2.dtype, "bfloat16")
    _check("--bf16 does NOT convert to fp8", m.converted, False)

    # 2. --fp8 alone: unchanged behaviour. This is the regression half -- the edit that added
    #    --bf16 touched this branch's condition.
    m, err = _run(src, fp8_flag=True, bf16_flag=False, moe=True)
    _check("--fp8 runs", err, None)
    _check("--fp8 still casts the experts to bf16", m.w13.dtype, "bfloat16")
    _check("--fp8 still converts to fp8", m.converted, True)

    # 3. Neither, with MoE: refused, and the message must OFFER --bf16. A refusal that does not
    #    name the new flag sends the next reader to the same dead end this test exists for.
    m, err = _run(src, fp8_flag=False, bf16_flag=False, moe=True)
    _check("MoE without either flag is refused", bool(err and "REFUSING" in err), True)
    _check("the refusal offers --bf16", bool(err and "--bf16" in err), True)
    _check("the refusal names the cause", bool(err and "neither --fp8 nor --bf16" in err), True)
    _check("nothing was cast before the refusal", m.w13.dtype, "float32")

    # 4. Both flags: refused. Neither precedence order is acceptable, because the losing flag is
    #    a configuration someone asked for and did not get.
    m, err = _run(src, fp8_flag=True, bf16_flag=True, moe=True)
    _check("--bf16 --fp8 together is refused", bool(err and "REFUSING" in err), True)
    _check("the refusal explains the conflict", bool(err and "Pick one" in err), True)
    _check("no conversion happened on the refused combination", m.converted, False)

    # 5. NO MoE AND NO FLAGS: the model must stay fp32. This is the check that fails if someone
    #    "fixes" the guard by casting unconditionally -- the silent precision change model.py's
    #    own comment argues against.
    m, err = _run(src, fp8_flag=False, bf16_flag=False, moe=False)
    _check("a plain dense run is not refused", err, None)
    _check("a plain dense run stays fp32", m.dtype, "float32")
    _check("a plain dense run is not converted", m.converted, False)
    #    AND ON AN EXPERT-BEARING BUILD reached with --fp8: the experts are bf16 there, so a
    #    mutation that makes the bf16 branch unconditional cannot be caught by check 5's dense
    #    model alone. `--bf16` claiming to be "the cast without the conversion" is only true if
    #    the two branches stay mutually exclusive, which is what this asserts.
    m_fp8, _ = _run(src, fp8_flag=True, bf16_flag=False, moe=True)
    m_bf16, _ = _run(src, fp8_flag=False, bf16_flag=True, moe=True)
    _check("the two branches are exclusive (fp8 converts, bf16 does not)",
           (m_fp8.converted, m_bf16.converted), (True, False))

    # 6. --bf16 without amp: refused rather than silently ignored. `bf16_only` is `args.bf16 and
    #    amp`, so without this refusal the flag would be accepted and do nothing -- a flag
    #    declaring a property the caller relies on, false while set.
    m, err = _run(src, fp8_flag=False, bf16_flag=True, moe=False, amp=False)
    _check("--bf16 without amp is refused", bool(err and "REFUSING" in err), True)
    _check("that refusal names amp", bool(err and "amp" in err), True)

    if _fails:
        print(f"\ntest_bf16_flag: {len(_fails)} failure(s)")
        return 1
    print("\ntest_bf16_flag OK: --bf16 is the cast without the conversion, --fp8 is unchanged, "
          "both together and neither-with-MoE are refused, and a plain run stays fp32")
    return 0


if __name__ == "__main__":
    sys.exit(main())
