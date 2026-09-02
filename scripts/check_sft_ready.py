#!/usr/bin/env python3
# restartable: reads two files and writes nothing. Every check is a comparison over metadata
# already on disk, so an interrupt costs a rerun of a few seconds, not work.
"""Is this (base checkpoint, SFT pack) pair ready to start SFT? CPU only, no cards.

    python scripts/check_sft_ready.py <ckpt> <pack>
    python scripts/check_sft_ready.py --selftest

e1-21's no-card half. Everything sft_math.py would discover in its first thirty seconds --
after allocating eight cards, building the model and loading the pack -- checked here on a
CPU in seconds. The three failures this exists to catch all train SILENTLY:

  a pack from another vocabulary trains at ~4x the loss with every id wrong and in range,
  because the sizes match (sft_math.py:110's own comment)

  a stale holdout fingerprint means the pack may hold held-out questions, and the run
  produces a checkpoint whose eval numbers are contaminated with no error anywhere

  a wrong loss mask trains without complaint, loses a couple of points, and nothing in the
  logs says why (scripts/test_sft_pack.py's own comment)

WHAT IT DELIBERATELY DOES NOT DO. It does not build the model, allocate a card, or read
anything outside the two paths given plus data/tokenizer.json and
data/eval/holdout_hashes.txt. On the pod the cards are running a resume and /data00 holds
token caches measured in tens of GB; a readiness check that competes with the job it is
preparing for is not a readiness check. Every load is mmap where torch allows it.

EXIT CODES: 0 ready; 1 a named check failed; 2 the check could not run (missing file,
unreadable pack) -- distinguished because "not ready" and "I could not tell" are different
facts and only the first is about the artifacts.
"""

import argparse
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "datagen"))

TOK = os.path.join(ROOT, "data", "tokenizer.json")
HOLDOUT = os.path.join(ROOT, "data", "eval", "holdout_hashes.txt")


def _load_meta(path):
    """Checkpoint metadata without pulling the weights off disk."""
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except Exception:  # noqa: BLE001 -- some runtimes reject mmap
        return torch.load(path, map_location="cpu", weights_only=False)


def check_vocab(ck, pack, out):
    """The pack's vocabulary must be the one the checkpoint was trained on.

    Compared on vocab_id, the same key both sides carry -- comparing vocab SIZE would pass
    for two different 32,784-slot vocabularies, which is the failure mode: every id valid,
    every id wrong.
    """
    ck_v, pk_v = ck.get("vocab_id"), pack.get("vocab_id")
    if ck_v is None and pk_v is None:
        out.append(("WARN", "neither the checkpoint nor the pack carries vocab_id; both "
                            "predate fingerprinting, so this cannot be checked -- verify "
                            "by hand which tokenizer built each"))
        return True
    if ck_v is None:
        out.append(("WARN", f"the checkpoint carries no vocab_id (pack has {pk_v}); pass "
                            f"--vocab to sft_math.py with the value from ckpt_info.py"))
        return True
    if pk_v is None:
        out.append(("FAIL", f"the pack carries no vocab_id but the checkpoint says {ck_v}; "
                            f"repack -- an unfingerprinted pack cannot be shown to match"))
        return False
    if ck_v != pk_v:
        out.append(("FAIL", f"vocabulary mismatch: checkpoint {ck_v}, pack {pk_v}. Every "
                            f"token id would be valid and wrong, and the loss would sit "
                            f"~4x high with no error. Repack against the checkpoint's "
                            f"tokenizer."))
        return False
    out.append(("ok", f"vocab_id matches on both sides: {ck_v}"))
    return True


def check_holdout(pack, out):
    """The pack's holdout fingerprint must match the live holdout set."""
    fp = pack.get("holdout_fp")
    if fp is None:
        out.append(("WARN", "the pack carries no holdout_fp; it predates holdout "
                            "fingerprinting and may contain held-out questions -- verify "
                            "by hand before trusting any eval taken after this SFT"))
        return True
    if not os.path.exists(HOLDOUT):
        out.append(("WARN", f"{os.path.relpath(HOLDOUT, ROOT)} absent, so the pack's "
                            f"holdout_fp {fp} cannot be compared"))
        return True
    live = hashlib.sha256(open(HOLDOUT, "rb").read()).hexdigest()[:16]
    if fp != live:
        out.append(("FAIL", f"holdout mismatch: pack built against {fp}, live set is "
                            f"{live}. The pack may hold questions the eval holds out, "
                            f"which contaminates every number taken after this SFT. "
                            f"Repack with datagen/prepare_sft.py."))
        return False
    out.append(("ok", f"holdout_fp matches the live set: {fp}"))
    return True


def check_sources_fp(pack, out):
    """Report the pack's own provenance rather than judging it.

    sources_fp used to be computed over prepare_sft's ten-file SOURCES no matter which
    packer ran, so a prepare_sft_math pack carried a fingerprint naming nine files it never
    read (fixed 2026-09-02, e1-21). A pack built before that fix carries the wrong list and
    there is no way to tell from the pack alone -- so this PRINTS the field and says what
    it cannot conclude, instead of comparing it to something.
    """
    fp = pack.get("sources_fp")
    packer = pack.get("packer_fp")
    if fp is None:
        out.append(("WARN", "no sources_fp: this pack does not record which files built it"))
        return True
    if fp == "caller-supplied examples, no source list":
        out.append(("ok", "sources_fp says the examples were built in-process (a test pack)"))
        return True
    out.append(("ok", f"sources_fp {fp}, packer_fp {packer} -- recorded, not verifiable "
                      f"from the pack alone. If this pack predates 2026-09-02 and came "
                      f"from prepare_sft_math.py, the field names the wrong source list."))
    return True


def check_mask(pack, out):
    """Prompt tokens masked, completion tokens supervised, padding masked.

    The pack's own labels, not a freshly packed fixture: test_sft_pack proves the PACKER is
    correct, and this has to prove THIS FILE is. A pack can be well-formed and still be the
    wrong artifact.
    """
    ids, lab = pack.get("input_ids"), pack.get("labels")
    if ids is None or lab is None:
        out.append(("FAIL", "the pack has no input_ids/labels; it is not an SFT pack"))
        return False
    if ids.shape != lab.shape:
        out.append(("FAIL", f"input_ids {tuple(ids.shape)} != labels {tuple(lab.shape)}"))
        return False
    n_rows, row_len = ids.shape
    sup = int((lab != -100).sum())
    total = int(lab.numel())
    if sup == 0:
        out.append(("FAIL", "every position is masked: this pack would train on nothing"))
        return False
    if sup == total:
        out.append(("FAIL", "no position is masked, so the prompt is supervised too -- the "
                            "model would be trained to generate the user's turn"))
        return False
    pct = 100.0 * sup / total
    # BAND MEASURED, NOT ASSUMED. The first version said (0.5, 60.0) on the reasoning that
    # "SFT rows are mostly prompt, so a few percent is normal and half is not". That reasoning
    # is wrong for this repo's data and the ceiling was never checked against a real pack: the
    # three packs behind be.sft_v3/v4/v5 all sit at 79.3% supervised, and so does the control
    # pack at 79.4% (measured on the pod 2026-09-03). Our prompts are short Chinese
    # instructions and the completions are long CoT answers, so most of every row IS the
    # answer. The old ceiling would have refused every genuine pack in the repo -- a gate
    # whose failing case is the normal artifact gets bypassed, and then it protects nothing.
    #
    # The ceiling that still means something is 100%: no masking at all, which check_mask
    # already catches above as its own named failure. This band keeps a floor (a pack that
    # supervises almost nothing is a mask defect) and a ceiling loose enough to admit the
    # measured population, 95%.
    band = (0.5, 95.0)
    verdict = "ok" if band[0] <= pct <= band[1] else "FAIL"
    out.append((verdict, f"{n_rows:,} rows x {row_len} tokens, {sup:,} supervised "
                         f"({pct:.1f}%), expected {band[0]}-{band[1]}%"))
    # Where labels are not -100 they must equal the input at that position: the packer
    # writes labels[t] = input_ids[t] for supervised tokens, so any other value is a
    # shifted or corrupted pack. Checked on a slice to stay cheap on a large pack.
    k = min(n_rows, 256)
    m = lab[:k] != -100
    if not bool((lab[:k][m] == ids[:k][m]).all()):
        out.append(("FAIL", "a supervised label differs from its input token: the pack is "
                            "shifted or corrupted (train.py slices x=[:, :-1], y=[:, 1:], "
                            "so the pack itself must be unshifted)"))
        return False
    out.append(("ok", f"every supervised label equals its input token (first {k} rows)"))
    return verdict == "ok"


def check_sft_math_parser(out):
    """sft_math.py's parser accepts the launch it will be given, and both grad_ckpt
    spellings reach the same effective value.

    Exec'd from the real source rather than imported: importing sft_math pulls torch, DDP
    and a checkpoint path. The region is the statements from ArgumentParser to parse_args,
    the same technique scripts/test_recipe_required.py uses on train.py, so a flag added to
    sft_math appears here with no edit.
    """
    import argparse as ap
    import ast
    import contextlib
    import io

    src = open(os.path.join(ROOT, "sft_math.py"), encoding="utf-8").read()
    tree = ast.parse(src)
    main = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"), None)
    if main is None:
        out.append(("FAIL", "sft_math.py has no top-level main()"))
        return False
    start = next((i for i, s in enumerate(main.body) if "ArgumentParser" in ast.unparse(s)), None)
    stop = next((i for i, s in enumerate(main.body) if "parse_args" in ast.unparse(s)), None)
    if start is None or stop is None or stop <= start:
        out.append(("FAIL", "could not delimit sft_math.py's parser region"))
        return False
    ns = {"argparse": ap, "SFT_DATA": "data/sft/sft_all.pt", "ROOT": ROOT, "os": os}
    try:
        exec(compile(ast.Module(body=main.body[start:stop], type_ignores=[]), "<p>", "exec"), ns)  # noqa: S102
    except Exception as e:  # noqa: BLE001
        out.append(("FAIL", f"sft_math.py's parser region needs more than argparse: "
                            f"{type(e).__name__}: {e}"))
        return False
    parser = ns.get("parser")
    if parser is None:
        out.append(("FAIL", "sft_math.py's parser region defined no `parser`"))
        return False

    def parse(argv):
        buf = io.StringIO()
        try:
            with contextlib.redirect_stderr(buf), contextlib.redirect_stdout(buf):
                return parser.parse_args(argv), None
        except SystemExit as e:
            return None, (e.code if e.code is not None else 0)

    ok = True
    cases = [
        (["--resume", "ck.pt"], True, "the minimal launch"),
        (["--resume", "ck.pt", "--no-grad_ckpt"], False, "--no-grad_ckpt (hyphen, train.py's spelling)"),
        (["--resume", "ck.pt", "--grad_ckpt"], True, "--grad_ckpt explicit"),
        (["--resume", "ck.pt", "--no_grad_ckpt"], False, "--no_grad_ckpt (deprecated underscore)"),
    ]
    for argv, want_gc, label in cases:
        args, code = parse(argv)
        if code is not None:
            out.append(("FAIL", f"sft_math.py refuses {label} (exit {code})"))
            ok = False
            continue
        eff = False if getattr(args, "no_grad_ckpt", False) else getattr(args, "grad_ckpt", None)
        if eff != want_gc:
            out.append(("FAIL", f"{label}: grad_ckpt is {eff}, expected {want_gc}"))
            ok = False
        else:
            out.append(("ok", f"parses, grad_ckpt={eff}: {label}"))
    # --resume is the one flag with no sane default: a missing base checkpoint must refuse.
    _, code = parse([])
    if code is None:
        out.append(("FAIL", "sft_math.py accepts a launch with no --resume"))
        ok = False
    else:
        out.append(("ok", f"omitting --resume is refused (exit {code})"))
    return ok


def selftest():
    """A synthetic pack for each failure this file exists to catch, then a good one.

    Every case asserts the NAMED check fires, not merely that something failed: a check
    that reports the wrong reason sends the reader to the wrong artifact.
    """
    import tempfile

    import torch

    def run(fn, *a):
        out = []
        passed = fn(*a, out)
        return passed, out

    fails = []

    # vocab mismatch
    ok, out = run(check_vocab, {"vocab_id": "aaa"}, {"vocab_id": "bbb"})
    if ok or not any("vocabulary mismatch" in m for _, m in out):
        fails.append(f"vocab mismatch not caught: {out}")
    ok, out = run(check_vocab, {"vocab_id": "aaa"}, {"vocab_id": "aaa"})
    if not ok:
        fails.append(f"matching vocab_id reported as a failure: {out}")

    # holdout mismatch
    ok, out = run(check_holdout, {"holdout_fp": "0" * 16})
    if os.path.exists(HOLDOUT) and (ok or not any("holdout mismatch" in m for _, m in out)):
        fails.append(f"holdout mismatch not caught: {out}")

    # masks
    ids = torch.arange(64, dtype=torch.int32).reshape(4, 16)
    lab_all = ids.clone()
    lab_none = torch.full_like(ids, -100)
    lab_good = ids.clone()
    lab_good[:, :12] = -100
    lab_shift = lab_good.clone()
    lab_shift[0, 13] = 999  # a supervised label that is not its input token
    for pack, must in (
        ({"input_ids": ids, "labels": lab_all}, "no position is masked"),
        ({"input_ids": ids, "labels": lab_none}, "every position is masked"),
        ({"input_ids": ids, "labels": lab_shift}, "shifted or corrupted"),
        ({"input_ids": ids}, "not an SFT pack"),
    ):
        ok, out = run(check_mask, pack)
        if ok or not any(must in m for _, m in out):
            fails.append(f"mask case {must!r} not caught: {out}")
    ok, out = run(check_mask, {"input_ids": ids, "labels": lab_good})
    if not ok:
        fails.append(f"a well-formed pack reported as a failure: {out}")

    # THE BAND MUST ADMIT THIS REPO'S REAL PACKS. The band shipped as (0.5, 60.0) and every
    # genuine pack here is at 79.3-79.4%, so the gate refused the normal artifact -- caught
    # only by running it on the pod, because every case above uses a synthetic fixture whose
    # supervised fraction I chose. A fixture at the measured rate is what makes this a check on
    # the BAND rather than on the arithmetic.
    real_rate_ids = torch.arange(1000, dtype=torch.int32).reshape(10, 100)
    real_rate_lab = real_rate_ids.clone()
    real_rate_lab[:, :21] = -100          # 79% supervised, the measured population
    ok, out = run(check_mask, {"input_ids": real_rate_ids, "labels": real_rate_lab})
    if not ok:
        fails.append(f"a pack at the repo's measured 79% supervised rate was refused: {out}")

    # the real parser
    ok, out = run(check_sft_math_parser)
    if not ok:
        fails.append(f"sft_math.py's own parser did not pass: {out}")

    # and end to end on a real temp pack, so the file's main path is exercised too
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "pack.pt")
        torch.save({"input_ids": ids, "labels": lab_good, "vocab_id": "xyz"}, p)
        loaded = torch.load(p, map_location="cpu", weights_only=True)
        ok, out = run(check_vocab, {"vocab_id": "xyz"}, loaded)
        if not ok:
            fails.append(f"round-tripped pack failed the vocab check: {out}")

    for f in fails:
        print(f"  SELFTEST FAIL {f}")
    if fails:
        print(f"\n{len(fails)} selftest failure(s)")
        return 1
    print("check_sft_ready selftest OK (every named check fires on its own broken pack)")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ckpt", nargs="?", help="base checkpoint to start SFT from")
    ap.add_argument("pack", nargs="?", help="SFT pack (.pt) to train on")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.ckpt or not a.pack:
        ap.error("need <ckpt> and <pack>, or --selftest")

    for p in (a.ckpt, a.pack):
        if not os.path.exists(p):
            print(f"CANNOT CHECK: {p} does not exist")
            return 2

    import torch

    try:
        ck = _load_meta(a.ckpt)
        pack = torch.load(a.pack, map_location="cpu", weights_only=True)
    except Exception as e:  # noqa: BLE001
        print(f"CANNOT CHECK: {type(e).__name__}: {e}")
        return 2
    if not isinstance(ck, dict) or not isinstance(pack, dict):
        print("CANNOT CHECK: checkpoint or pack is not a dict")
        return 2

    print(f"ckpt {a.ckpt}  step {ck.get('step')}  cfg layers "
          f"{ck.get('cfg', {}).get('layers')} d {ck.get('cfg', {}).get('d')}")
    print(f"pack {a.pack}\n")

    out = []
    passed = all([
        check_vocab(ck, pack, out),
        check_holdout(pack, out),
        check_sources_fp(pack, out),
        check_mask(pack, out),
        check_sft_math_parser(out),
    ])
    for verdict, msg in out:
        print(f"  {verdict:4} {msg}")
    n_fail = sum(1 for v, _ in out if v == "FAIL")
    n_warn = sum(1 for v, _ in out if v == "WARN")
    print(f"\n{n_fail} FAIL, {n_warn} WARN, {sum(1 for v, _ in out if v == 'ok')} ok")
    if not passed:
        print("NOT READY: fix the FAIL lines above before starting SFT.")
        return 1
    if n_warn:
        print("READY, with warnings: each WARN names something this check could not "
              "verify, not something it verified as fine.")
    else:
        print("READY: vocabulary, holdout, mask and launch parser all check out.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
