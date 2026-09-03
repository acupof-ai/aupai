#!/usr/bin/env python3
# restartable: one JSON line per problem is appended to --preds as it is scored, and a rerun with
# the same --preds skips ids already there. An interrupt costs the current problem only.
"""math_bpb: bits per byte of the math_test_500 gold solutions, teacher-forced.

    python3 eval/math_bpb.py --ckpt <ckpt.pt>
    python3 eval/math_bpb.py --ckpt <hf-dir> --hf      # control arm, its own tokenizer
    python3 eval/math_bpb.py --selftest                # no card, no model

N3's second metric (roadmap_0903.md, owner e1, reviewer b0). The scoring machinery is IMPORTED
from eval/humaneval_bpb.py, not re-implemented: solution_bpb, OursModel and HFModel are the
same functions, so the two metrics cannot drift in their context handling, their bit conversion,
or their byte divisor. A second copy of solution_bpb is how humaneval and math would end up
reporting bits/byte computed two ways under one name -- the defect this repo has recorded for
predicates (§145) and for guards sharing an implementation with their test (§153).

WHY THIS DATASET AND NOT GSM8K, from eval_resolution_200m.md §Ranked additions 2, verified here
rather than inherited: OLMo 3 §3.3.2 routes math BPB through Minerva's human-written solutions
and reports Minerva BPB SNR 88.6 against GSM8K BPB 7.0, the likely cause being GSM8K's
calculator annotations sitting out-of-distribution for a base model. Our golds are
Minerva-shaped: measured on data/eval/math_test_500.jsonl, 0 of 500 rows contain `<<`, 494 of
500 end in a `\\boxed{}`, and the golds total 381,669 UTF-8 bytes. So the objection that
excludes GSM8K does not apply to us.

WHAT IS SCORED. `instruction` is context and carries NO loss; `output` (the full worked solution)
is the target. Same split as humaneval_bpb, same reason: scoring the prompt too would mix "can
it model a Chinese word problem" into a number reported as math modelling.

WHY PER BYTE. Cross-tokenizer honesty. Ours is 32,773 entries, the control's NeoX BPE is 50,304,
and Chinese text goes through byte fallback on the control side -- so per-token loss compares
two different quantities while UTF-8 bytes are identical on both sides (1e's standing ruling).

BOUNDARY, and it is the same one humaneval_bpb carries: BPB says the model assigns the real
solution higher likelihood. It does NOT say the model would produce it. The generative reading on
these very problems is floored -- e1-31/e1-31b measured 27/497 = 5.4% and 17/497 = 3.4% for our
arm against 0/497 for the control, every cell under the 12.6pt threshold, all read as zero. A
falling BPB beside a floored accuracy is exactly the split
be.gold_bpb_falls_while_generation_scores_zero records, and the two must not be read as one.
"""

import argparse
import importlib.util
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("FLA_FLASH_KDA", "0")

# IMPORTED BY PATH because eval/ has no __init__ and humaneval_bpb sets FLA_FLASH_KDA at import.
_spec = importlib.util.spec_from_file_location(
    "humaneval_bpb", os.path.join(ROOT, "eval", "humaneval_bpb.py"))
_H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_H)
solution_bpb, OursModel, HFModel = _H.solution_bpb, _H.OursModel, _H.HFModel

TOK_PATH = os.path.join(ROOT, "data", "tokenizer.json")
DEFAULT_DATA = os.path.join(ROOT, "data", "eval", "math_test_500.jsonl")
N_PROBLEMS = 500        # the file's own length; a short file is a truncated copy, not a small run
GOLD_BYTES = 381_669    # measured 2026-09-03; a different total means a different file
N_DEMOS_EXCLUDED = 3    # rows 0..2 are l1_fewshot's demos -- see load_problems


def load_problems(path, limit=None, skip_demos=True):
    """(id, instruction, output) per row; a row missing either field REFUSES.

    Refuses rather than skips for the same reason humaneval_bpb does: the denominator is the
    problem count, and silently scoring 499 of 500 makes two runs incomparable while both look
    complete.

    SKIPS ROWS 0..2 BY DEFAULT. Those three are the demos l1_fewshot puts in every prompt
    (eval/l1_fewshot.py's split_rows takes rows 0..n-1 as demos), so a checkpoint evaluated after
    any few-shot run has seen them as CONTEXT, and a checkpoint whose SFT pack drew from this
    file has seen them as targets. Keeping them would put three rows of known-exposure text into
    a metric whose whole job is comparing checkpoints. --keep_demos scores all 500 for anyone who
    wants the full-file figure, and the summary records which was used.
    """
    out = []
    with open(path, encoding="utf-8") as f:
        for k, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            for field in ("instruction", "output"):
                if not r.get(field):
                    raise ValueError(f"{path} row {k} has no {field!r}; this file cannot be "
                                     f"scored without it")
            if skip_demos and k < N_DEMOS_EXCLUDED:
                continue
            out.append((f"math_test_500/{k}", r["instruction"], r["output"]))
            if limit and len(out) >= limit:
                break
    return out


def _selftest():
    """Only what is NOT already covered by humaneval_bpb --selftest.

    solution_bpb's arithmetic (log2(V) on a uniform model, prompt excluded, byte divisor,
    over-context trimming) is tested there, on the same function this file imports. Re-testing it
    here would test a second copy of the assertions against one implementation, which proves the
    fixture and not the code. What is new here is the DATASET contract, so that is what this
    checks -- plus one assertion that the import really is the shared function.
    """
    assert solution_bpb is _H.solution_bpb, "solution_bpb is not humaneval_bpb's function"
    _H._selftest()   # the shared arithmetic, run against the shared implementation

    import tempfile
    rows = [{"instruction": f"q{i}", "output": f"a{i}"} for i in range(6)]
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as tf:
        for r in rows:
            tf.write(json.dumps(r, ensure_ascii=False) + "\n")
        good = tf.name
    try:
        # THE DEMOS ARE EXCLUDED BY DEFAULT, and included only when asked.
        got = load_problems(good)
        assert [i for i, _, _ in got] == [f"math_test_500/{k}" for k in (3, 4, 5)], got
        allrows = load_problems(good, skip_demos=False)
        assert len(allrows) == 6, allrows
        # THE ID CARRIES THE FILE ROW, so a --keep_demos run and a default run that both score
        # row 4 agree on its name; ids built from a position in the FILTERED list would not.
        assert allrows[4][0] == "math_test_500/4", allrows[4]
    finally:
        os.unlink(good)

    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as tf:
        tf.write(json.dumps({"instruction": "q", "output": "a"}) + "\n")
        tf.write(json.dumps({"instruction": "q"}) + "\n")
        bad = tf.name
    try:
        try:
            load_problems(bad, skip_demos=False)
            raise AssertionError("a row with no output was accepted")
        except ValueError as e:
            assert "output" in str(e), e
    finally:
        os.unlink(bad)

    # THE REAL FILE'S CONTRACT, if it is present: the two facts that promoted this metric over
    # GSM8K. A file that has drifted from these is a different dataset wearing the same path.
    if os.path.exists(DEFAULT_DATA):
        raw = [json.loads(x) for x in open(DEFAULT_DATA, encoding="utf-8") if x.strip()]
        assert len(raw) == N_PROBLEMS, f"{DEFAULT_DATA} has {len(raw)} rows, expected {N_PROBLEMS}"
        total = sum(len(r["output"].encode("utf-8")) for r in raw)
        assert total == GOLD_BYTES, (
            f"gold bytes {total} != {GOLD_BYTES} recorded 2026-09-03. The metric's divisor "
            f"changed, so figures across this change are not comparable -- re-measure and "
            f"update the constant in the same commit that explains why the file moved.")
        annotated = [k for k, r in enumerate(raw) if "<<" in r["output"]]
        assert not annotated, (
            f"rows {annotated[:5]} carry GSM8K-style `<<...>>` calculator annotations. Those "
            f"are what makes GSM8K BPB read SNR 7.0 against Minerva's 88.6, and their absence "
            f"is the measured reason this dataset was promoted (eval_resolution_200m.md).")

    print("math_bpb self-test OK: the arithmetic is humaneval_bpb's own function and its "
          "self-test passes here; demos 0-2 are excluded by default and ids carry the FILE row; "
          "a row missing a field refuses; and the real file still has 500 rows, 381,669 gold "
          "bytes and zero `<<...>>` annotations")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", help="our .pt, or an HF directory with --hf")
    ap.add_argument("--hf", action="store_true", help="control arm: HF format, own tokenizer")
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--tokenizer", default=TOK_PATH)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, help="first N problems (smoke test)")
    ap.add_argument("--max_ctx", type=int, default=2048)
    ap.add_argument("--keep_demos", action="store_true",
                    help="score all 500 rows including l1_fewshot's three demos (default: skip "
                         "them, because a few-shot-evaluated checkpoint has seen them as context)")
    ap.add_argument("--preds", help="jsonl, appended per problem; rerun resumes from it")
    ap.add_argument("--out", help="summary json")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        _selftest()
        return 0
    if not a.ckpt:
        ap.error("--ckpt required (or --selftest)")
    if not os.path.exists(a.data):
        sys.exit(f"math_test_500 missing: {a.data}")

    problems = load_problems(a.data, a.limit, skip_demos=not a.keep_demos)
    expected = N_PROBLEMS - (0 if a.keep_demos else N_DEMOS_EXCLUDED)
    if not a.limit and len(problems) != expected:
        sys.exit(f"{a.data} yielded {len(problems)} problems, expected {expected}. A short file "
                 f"is a truncated copy; scoring it would produce a number that looks like this "
                 f"metric and is not.")

    m = HFModel(a.ckpt, a.device) if a.hf else OursModel(a.ckpt, a.tokenizer, a.device)
    print(f"Loaded {a.ckpt}: {m.n_params / 1e6:.2f}M params | problems {len(problems)}", flush=True)

    done = {}
    if a.preds and os.path.exists(a.preds):
        with open(a.preds, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                done[r["id"]] = r
        print(f"resuming: {len(done)} problems already in {a.preds}", flush=True)

    vals, errs = [], []
    for pid, instruction, gold in problems:
        r = done.get(pid)
        if r is None:
            bpb, err = solution_bpb(m, instruction, gold, a.max_ctx)
            r = {"id": pid, "bpb": bpb, "error": err,
                 "n_bytes": len(gold.encode("utf-8"))}
            if a.preds:
                with open(a.preds, "a", encoding="utf-8") as f:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    f.flush()
        if r.get("bpb") is None:
            errs.append((r["id"], r.get("error")))
        else:
            vals.append((r["bpb"], r["n_bytes"]))
        if len(vals) % 100 == 0 and len(vals):
            print(f"  {len(vals)}/{len(problems)}", flush=True)

    if not vals:
        print("REFUSING: no problem produced a number")
        return 1
    # TWO means, as in humaneval_bpb: the per-problem mean weights every problem equally, the
    # byte-weighted mean is the bits/byte of the corpus as one string. They differ when solution
    # lengths differ, and here they differ a lot -- the golds run from tens to thousands of bytes.
    per_problem = sum(v for v, _ in vals) / len(vals)
    tot_bytes = sum(n for _, n in vals)
    byte_weighted = sum(v * n for v, n in vals) / tot_bytes

    result = {
        "ckpt": a.ckpt, "hf": a.hf, "n_problems": len(vals), "n_problems_total": len(problems),
        "demos_excluded": not a.keep_demos, "n_params": m.n_params,
        "math_bpb_per_problem_mean": per_problem,
        "math_bpb_byte_weighted": byte_weighted,
        "total_gold_bytes": tot_bytes,
        "errors": [{"id": t, "error": e} for t, e in errs],
        "reading": "bits per UTF-8 byte of the math_test_500 gold solution, teacher-forced, the "
                   "instruction carries no loss; cross-tokenizer comparable by construction",
        "boundary": "NOT an accuracy substitute: this says the real solution gets higher "
                    "likelihood, not that the model would produce it. The generative reading on "
                    "these same problems is FLOORED -- e1-31b, shared decoder: ours 5.4%/3.4%, "
                    "control 0.0%/0.0%, every cell under the 12.6pt threshold and all read as "
                    "zero. Report this beside that floor, never instead of it.",
    }
    if errs:
        result["boundary"] += (f" {len(errs)} of {len(problems)} problems produced no number and "
                               f"are listed in `errors`; the means are over the rest.")
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(result, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(json.dumps(result, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
