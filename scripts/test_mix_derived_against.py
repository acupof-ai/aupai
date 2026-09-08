#!/usr/bin/env python3
"""A resume mix names the cursor state it was derived against, and a different one refuses.

    python3 scripts/test_mix_derived_against.py

WHAT THIS IS FOR. `scripts/write_mix_500m.py --resume-cursor CKPT` makes every `epochs` value a
TOTAL -- cursor + this plan -- so the file is correct for exactly one resume point and silently
wrong for any other. Until `_derived_against` nothing in the file said which point, and a field
nobody compares is a comment, so the writer and the reader are asserted together here.

THE PROPERTY IS THE TRIPLE, NOT A FILE IDENTITY (4c's ruling 2026-09-07). Two checkpoints with
equal cursor state are substitutable and case 5 asserts they pass; a path or a hash would refuse
one of them for no reason. Rows alone are not the property either: cases 3 and 4 hold the rows
fixed and move the corpus fingerprint and the sample seed, because the same row numbers over a
re-fingerprinted or re-shuffled corpus name different documents.

THE READER IS EXERCISED THROUGH build_mix's OWN CALL, not by calling the helper with hand-built
dicts. A test that calls _assert_mix_derived_against directly would pass even if build_mix never
called it -- which is exactly the defect the field exists to avoid, one level up.
"""

import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

FAILS = []

CUR = {"cot": 290_401, "code_rp1t": 1_338_744}
FP = {"cot": "388496b76ed9bf88", "code_rp1t": "d8b9b18ba080f487"}
SEED = 0


def check(ok, msg):
    if not ok:
        FAILS.append(msg)


def _mix(tmp, derived):
    """A real mix shape, with only _derived_against varying between worlds."""
    m = {
        "total_tokens": 8192 * 100,
        "seq": 8192,
        "domains": {n: {"weight": 0.5, "epochs": 4, "anneal": 0.5} for n in CUR},
    }
    if derived is not None:
        m["_derived_against"] = derived
    p = os.path.join(tmp, "mix.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(m, f)
    return p


def _run(train, path, cursor, fp, seed):
    """build_mix far enough to reach the guard, as one of three outcomes.

    tok=None touches no cache, and the call dies further down on a missing corpus, so "the
    guard did not fire" and "the guard crashed" both arrive as an exception and must not be
    folded together. Folding them was a real defect in this file's first version: a mutant
    that made the row comparison raise KeyError was reported as SURVIVED, because the crash
    was caught and returned as "passed". Three outcomes, and the caller asserts on which:
      the refusal string -- the guard refused
      None              -- the guard passed and the call died later, as expected
      a "CRASH: ..." string -- something raised inside the guard
    """
    try:
        train.build_mix(path, None, False, False, row_cursor=cursor, cursor_srcfp=fp, cursor_seed=seed)
    except RuntimeError as e:
        if "derived against" in str(e):
            return str(e)
        return None
    except Exception as e:
        # Only a raise from _assert_mix_derived_against itself counts; anything downstream is
        # the corpus this fixture does not have. A crash inside the guard is recorded as a
        # FAILURE here rather than returned for the caller to interpret: `got is not None`
        # reads a crash as a refusal, which is how removing the fresh-start branch went green
        # -- the removed raise became an AttributeError on `None.items()` two lines down.
        import traceback

        if "_assert_mix_derived_against" in "".join(traceback.format_exc()):
            FAILS.append(f"the guard itself raised {type(e).__name__}: {e}")
            return f"CRASH: {type(e).__name__}: {e}"
        return None
    return None


def main():
    import train

    tmp = tempfile.mkdtemp(prefix="derived_")
    full = {"row_cursor": CUR, "row_cursor_srcfp": FP, "row_cursor_seed": SEED}

    # 1. THE MATCH PASSES. Without this every later refusal could be the guard refusing
    #    everything, which is a check that expresses nothing.
    got = _run(train, _mix(tmp, full), dict(CUR), dict(FP), SEED)
    check(got is None, f"an exactly-matching triple was refused: {got}")

    # 2. ROWS DIFFER -> REFUSE. 4c's acceptance test, verbatim: a mix derived at rows R against
    #    a checkpoint at R' != R.
    moved = dict(CUR, cot=CUR["cot"] + 1)
    got = _run(train, _mix(tmp, full), moved, dict(FP), SEED)
    check(got is not None, "a cursor 1 row past the mix's was accepted")
    check(
        got and "cot" in got and "290401" in got.replace(",", ""),
        f"the refusal must name the domain and both numbers: {got}",
    )

    # 3. EQUAL ROWS, CHANGED FINGERPRINT -> REFUSE. The half a row comparison cannot see: the
    #    same prefix length over a different corpus is a different set of documents.
    got = _run(train, _mix(tmp, full), dict(CUR), dict(FP, cot="0000000000000000"), SEED)
    check(got is not None, "equal rows against a changed corpus fingerprint were accepted")
    check(got and "fingerprint" in got, f"the refusal must say what changed: {got}")

    # 4. EQUAL ROWS, CHANGED SEED -> REFUSE. Same reasoning, different mechanism: the shuffle
    #    order decides which documents the first N rows are.
    got = _run(train, _mix(tmp, full), dict(CUR), dict(FP), SEED + 1)
    check(got is not None, "equal rows against a changed sample seed were accepted")
    check(got and "seed" in got, f"the refusal must say what changed: {got}")

    # 5. AN EQUAL TRIPLE FROM A DIFFERENT CHECKPOINT PASSES. The ruling's other half: cursor
    #    state is the identity, so a substitutable checkpoint must not be refused. Same values,
    #    fresh dict objects -- if this ever fails on object identity the guard is comparing the
    #    wrong thing.
    got = _run(train, _mix(tmp, full), {k: v for k, v in CUR.items()}, {k: v for k, v in FP.items()}, SEED)
    check(got is None, f"an equal triple from a different checkpoint was refused: {got}")

    # 6. A MIX THE CHECKPOINT DOES NOT COVER -> REFUSE. A domain the mix claims and the
    #    checkpoint lacks is a claim about a prefix that does not exist.
    got = _run(train, _mix(tmp, full), {"cot": CUR["cot"]}, dict(FP), SEED)
    check(got is not None, "a mix naming a domain the checkpoint's cursor lacks was accepted")

    # 7. THE OPPOSITE DIRECTION PASSES, and it is not symmetry for its own sake:
    #    mix_30b_stage2 renames en_c4 -> en_c4_stage2, so its cursor names 5 domains against a
    #    checkpoint's 7. Dict equality would refuse the one resume that file was written for.
    #    THE EXTRA DOMAIN CARRIES A FINGERPRINT AND THE MIX DOES NOT MENTION IT, so this also
    #    discriminates a srcfp comparison widened to the union: intersect passes, union refuses.
    got = _run(train, _mix(tmp, full), dict(CUR, en_c4=755_274), dict(FP, en_c4="05e0fc6f14704056"), SEED)
    check(got is None, f"a checkpoint carrying an extra domain was refused: {got}")

    # 8. NO FIELD -> NO OPINION. All 24 mixes committed before this carry nothing. ASSERTED
    #    WITH A CURSOR THAT WOULD OTHERWISE REFUSE ON ALL THREE HALVES -- rows, fingerprint and
    #    seed each moved. A guard that treats an absent field as an empty claim refuses here;
    #    reading the file's silence as silence is the only way this passes.
    got = _run(train, _mix(tmp, None), {"cot": 1}, {"cot": "ffffffffffffffff"}, SEED + 99)
    check(got is None, f"a mix with no _derived_against was refused: {got}")

    # 8b. NO FIELD AND NO CURSOR -> SILENT. The combination every from-scratch run on all 24
    #     committed mixes actually has, and the one 8 above cannot cover: with a cursor present
    #     the fresh-start branch is never reached, so a guard that defaults an absent field to
    #     an empty claim passes 8 and refuses every real fresh start. Measured: substituting
    #     `da = mix.get(...) or {"row_cursor": {}}` leaves 8 green and fails only here.
    got = _run(train, _mix(tmp, None), None, None, None)
    check(
        got is None,
        f"a fieldless mix on a fresh start was refused -- that is every from-scratch run on "
        f"all 24 committed mixes: {got}",
    )

    # 9. A RESUME MIX ON A FRESH START -> REFUSE. The direction that loses data rather than
    #    raising: those `epochs` are totals that already count rows this run would draw again.
    got = _run(train, _mix(tmp, full), None, None, None)
    check(got is not None, "a resume-derived mix was accepted for a fresh start")

    # 10. PARTIAL FIELDS COMPARE ONLY WHAT IS THERE. write_mix_stage2 transcribes a cursor
    #     constant and reads no checkpoint, so it states rows and omits srcfp and seed;
    #     inventing a seed there would refuse every real checkpoint.
    got = _run(train, _mix(tmp, {"row_cursor": CUR}), dict(CUR), dict(FP), SEED + 7)
    check(got is None, f"a rows-only _derived_against was refused over a seed it never claimed: {got}")

    # 11. AND A ROWS-ONLY FIELD STILL REFUSES ON ROWS -- otherwise case 10 would have bought
    #     the refusal off entirely.
    got = _run(train, _mix(tmp, {"row_cursor": CUR}), moved, dict(FP), SEED)
    check(got is not None, "a rows-only _derived_against ignored a row mismatch")

    # 12. THE WRITER EMITS THE FIELD, asserted on write_mix_500m's OWN SOURCE rather than by
    #     running it -- main() needs the corpus stamps, which live on the pod, so a run here
    #     refuses before it writes anything. Without this arm the whole file passes with the
    #     writer emitting nothing: every case above builds its own mix, so the reader would be
    #     guarding a field no producer ever sets. Source-level, and narrow: the key must be
    #     assigned in the writer under the cursor branch, and _read_cursor must hand back all
    #     three parts for it to be assignable at all.
    w = open(os.path.join(ROOT, "scripts", "write_mix_500m.py"), encoding="utf-8").read()
    check(
        'm["_derived_against"]' in w,
        "write_mix_500m.py assigns no m['_derived_against'], so no mix it writes carries the "
        "field and the reader above guards nothing",
    )
    for part in ("row_cursor", "row_cursor_srcfp", "row_cursor_seed"):
        check(
            f'"{part}":' in w.split('m["_derived_against"]')[-1][:600],
            f"write_mix_500m's _derived_against does not state {part}, so the reader cannot "
            f"compare it and that half of the triple is unenforced",
        )
    check(
        "cursor, cursor_srcfp, cursor_seed = _read_cursor" in w,
        "_read_cursor's three-part return is not unpacked in write_mix_500m.main, so srcfp "
        "and seed are discarded again and the field cannot state them",
    )

    # 13. THE SHRINKING --total REFUSAL, RUN rather than grepped. b0-34's original defect: the
    #     branch copies weights from a cursor-FREE reference build, so the cursor-aware sizing
    #     is discarded while _launch_blocked, computed against the cursor, stays in the file.
    #
    #     BOTH DIRECTIONS, because a refusal that fires on everything is not this refusal. The
    #     raising arm must get PAST this check -- it dies further down on the corpus stamps,
    #     which live on the pod, and that different failure is the evidence the check let it
    #     through. Asserted on the message, not just on the rc: argparse exits 2 for any error.
    import subprocess

    import torch

    ckpt = os.path.join(tmp, "cursor.pt")
    torch.save(
        {
            "row_cursor": {"cot": CUR["cot"]},
            "row_cursor_basis": "full_plan_prefix",
            "row_cursor_srcfp": {"cot": FP["cot"]},
            "row_cursor_seed": SEED,
        },
        ckpt,
    )
    for total, expect_refusal in (("8e9", True), ("30e9", False)):
        r = subprocess.run(
            [
                sys.executable,
                os.path.join(ROOT, "scripts", "write_mix_500m.py"),
                "--total",
                total,
                "--out",
                os.path.join(tmp, "out.json"),
                "--resume-cursor",
                ckpt,
            ],
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        hit = "--resume-cursor was passed" in (r.stderr + r.stdout)
        if expect_refusal:
            check(
                hit,
                f"--total {total} with --resume-cursor was NOT refused: "
                f"rc={r.returncode} {(r.stderr or r.stdout)[-200:]}",
            )
        else:
            check(
                not hit,
                f"--total {total} is ABOVE the default, where weights are recomputed "
                f"under the cursor, and the shrinking refusal fired anyway",
            )

    # 14. AND write_mix_stage2 STATES ITS OWN CURSOR, or the reader is vacuous on the one
    #     committed mix that actually has a cursor footprint.
    s = open(os.path.join(ROOT, "scripts", "write_mix_stage2.py"), encoding="utf-8").read()
    check(
        '"_derived_against"' in s,
        "write_mix_stage2.py writes no _derived_against, so mix_30b_stage2 -- the committed "
        "mix whose epochs ARE cursor-seeded -- claims nothing and the guard cannot fire on it",
    )

    # 15. _read_cursor HANDS BACK ALL THREE PARTS, called on a real checkpoint. Case 12 greps
    #     the unpack and a rows-only return satisfies it -- the tuple still unpacks, srcfp comes
    #     back {} and seed None, and every mix would then state a triple with two empty thirds
    #     that the reader compares against nothing. That was the shape this function had before
    #     b0-34, so it is the regression most likely to come back.
    import scripts.write_mix_500m as w500

    rows, fp, seed = w500._read_cursor(ckpt)
    check(rows == {"cot": CUR["cot"]}, f"_read_cursor lost the rows: {rows}")
    check(
        fp == {"cot": FP["cot"]},
        f"_read_cursor discarded row_cursor_srcfp: {fp} -- the mix would then claim an empty "
        f"fingerprint map and the corpus half of the triple would compare nothing",
    )
    check(
        seed == SEED,
        f"_read_cursor discarded row_cursor_seed: {seed!r} -- the mix would claim None and the "
        f"reader skips a key that is absent, so the shuffle half would go unenforced",
    )

    for f in FAILS:
        print(f"FAIL: {f}")
    if FAILS:
        return 1
    print(
        "ok  the triple is compared, an equal triple from another checkpoint passes, a "
        "changed fingerprint or seed at equal rows refuses, and an absent field is silent"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
