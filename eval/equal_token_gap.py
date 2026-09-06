#!/usr/bin/env python3
"""Equal-token val gap: MoE-48 at 8B tokens against two 0.2b-batch comparators (e1-42).

THE QUESTION. b0_moe48_8b runs at global batch 192 (8 x accum 4 x 6 cards), so 786,432 tokens
per step; the two comparators run at global batch 64 (16 x accum 2 x 2 cards), 262,144 per step.
Three times the tokens per step, so MoE step N sees the same tokens as a comparator's step 3N,
and only that pairing compares val at MATCHED TOKENS rather than at matched steps. Verified
against the logs' own counters rather than assumed: the dense arm prints 0.16B / 0.31B / 0.47B /
0.63B / 0.79B at steps 600 / 1200 / 1800 / 2400 / 3000, which is what MoE steps 200 / 400 / 600 /
800 / 1000 print.

TWO COMPARATORS, TWO DIFFERENT FACTS, and separating them is the point of this script:

  b0_e1p_dense   -- no moe_* keys in its cfg line. The architecture comparison the task title
                    asks for, and the confounds are architecture AND batch AND warmup together.
  b0_e1p_moe48   -- moe_arm='e1p_moe48', moe_experts=48. SAME architecture, so the confounds
                    are batch and warmup ALONE. This is the sharper of the two.

WHY THE SCRIPT EXISTS RATHER THAN A HAND-COPIED TABLE. The five gaps first handed to me as
"MoE vs dense" -- 0.740, 0.175, 0.082, 0.067, 0.066 -- are the MoE-vs-MoE curve, exactly, and the
dense curve is 0.668, 0.140, 0.052, 0.037, 0.038. Both sets are real and they differ by 0.03-0.07
at every point, which is larger than the effect anyone would read off either. I found the mixup by
asking which comparator WOULD produce those gaps (required val 2.725 / 2.501 / 2.383 / 2.317 /
2.255) and grepping every log on the pod for those five values; only b0_e1p_moe48.log held all
five. So the KNOWN_ANSWERS below pin both sets against their own arm: a future run of this script
on the resumed log cannot silently relabel one curve as the other.

THE LOGS LIVE ON THE POD, not in the repo: /work/aupai/runs/<name>.log, read through ~/bin/pod.
b0_moe48_8b was still `running` when this was written (step 1000 of 10172, 5 val points), and its
resume writes to the same file, so re-running this script after the resume extends the table --
the known-answer world keeps the five landed pairs honest while it grows.

    python3 eval/equal_token_gap.py --selftest     # known answers, no pod access
    python3 eval/equal_token_gap.py                # reads the pod's logs, prints both tables
"""

import argparse
import os
import re
import subprocess
import sys

POD = os.path.expanduser("~/bin/pod")
POD_RUNS = "/work/aupai/runs"

ARM = "b0_moe48_8b"
# name -> (label, what the confound list says)
COMPARATORS = {
    "b0_e1p_dense": "dense 0.2b",
    "b0_e1p_moe48": "MoE-48 at 0.2b batch",
}

# tok/step, from the cfg line: batch x accum x cards x seq. Asserted against the log, never
# assumed -- a comparator launched at a different batch would silently break the 3N pairing.
TOK_PER_STEP = {ARM: 8 * 4 * 6 * 4096, "b0_e1p_dense": 16 * 2 * 2 * 4096,
                "b0_e1p_moe48": 16 * 2 * 2 * 4096}

VAL_RE = re.compile(r"^step (\d+)/(\d+) val ([\d.]+)")

# The landed pairs, per comparator: {moe_step: gap}. Both sets are pinned because the mixup this
# script exists to prevent is exactly using one arm's numbers under the other's name.
KNOWN_ANSWERS = {
    "b0_e1p_dense": {200: 0.668, 400: 0.140, 600: 0.052, 800: 0.037, 1000: 0.038},
    "b0_e1p_moe48": {200: 0.740, 400: 0.175, 600: 0.082, 800: 0.067, 1000: 0.066},
}
KNOWN_ARM_VAL = {200: 3.465, 400: 2.676, 600: 2.465, 800: 2.384, 1000: 2.321}


def pod_read(name):
    """The val lines and the cfg line of <name>.log on the pod, as text.

    One pod call per log: ~/bin/pod re-parses its argument, so the command stays a plain grep
    with no parentheses or quoted prose (pod argv cannot carry prose).
    """
    cmd = (f"grep -E '^step [0-9]+/[0-9]+ val ' {POD_RUNS}/{name}.log; "
           f"grep -m1 'cfg batch' {POD_RUNS}/{name}.log")
    r = subprocess.run([POD, cmd], capture_output=True, text=True, timeout=300)
    if r.returncode != 0 and not r.stdout.strip():
        raise RuntimeError(f"pod read of {name}.log failed: {(r.stderr or '')[:200]}")
    return r.stdout


def parse_val(text):
    """{step: val} from log text. Last write wins: a resumed run reprints steps it redoes."""
    out = {}
    for line in text.splitlines():
        m = VAL_RE.match(line.strip())
        if m:
            out[int(m.group(1))] = float(m.group(3))
    return out


def parse_tok_per_step(text, cards):
    """tok/step read off the cfg line, so the pairing rests on the log and not on my arithmetic.

    Returns None when no cfg line is present (the selftest's fixtures carry only val lines).
    """
    m = re.search(r"cfg batch (\d+) accum (\d+) seq (\d+)", text)
    if not m:
        return None
    b, acc, seq = (int(x) for x in m.groups())
    return b * acc * cards * seq


def pairs(arm_val, cmp_val, ratio):
    """[(moe_step, tokens, arm_val, cmp_val, gap)] at matched tokens.

    Only steps where BOTH sides have a val point: the comparator stops at its own max step, and
    silently dropping the arm's later points would shorten the table without saying so.
    """
    out = []
    for s in sorted(arm_val):
        cs = s * ratio
        if cs not in cmp_val:
            continue
        out.append((s, s * TOK_PER_STEP[ARM], arm_val[s], cmp_val[cs],
                    round(arm_val[s] - cmp_val[cs], 4)))
    return out


def _selftest():
    """Known answers on both comparators, plus the failure the real data taught.

    The fixtures are the real val curves, so this is a regression test on the pairing and the
    arithmetic, not on invented numbers.
    """
    arm = {200: 3.465, 400: 2.676, 600: 2.465, 800: 2.384, 1000: 2.321}
    dense = {600: 2.797, 1200: 2.536, 1800: 2.413, 2400: 2.347, 3000: 2.283,
             200: 3.904, 400: 3.081, 800: 2.673, 1000: 2.590}
    moe_small = {600: 2.725, 1200: 2.501, 1800: 2.383, 2400: 2.317, 3000: 2.255,
                 200: 3.934, 400: 2.933, 800: 2.616, 1000: 2.547}

    n = 0
    for name, cmp_val in (("b0_e1p_dense", dense), ("b0_e1p_moe48", moe_small)):
        got = {s: g for s, _t, _a, _c, g in pairs(arm, cmp_val, 3)}
        want = KNOWN_ANSWERS[name]
        assert set(got) == set(want), f"{name}: paired {sorted(got)}, expected {sorted(want)}"
        for s in want:
            assert abs(got[s] - want[s]) < 5e-4, \
                f"{name} step {s}: gap {got[s]:+.4f}, known answer {want[s]:+.4f}"
        n += len(want)
    print(f"  ok   {n} known pairs reproduce across both comparators")

    # THE TWO CURVES MUST NOT BE INTERCHANGEABLE, and this is the assertion that catches the
    # mixup the script was written for: the five dense gaps against the MoE comparator's known
    # answers must FAIL. Without this the selftest passes for a script that reads either log
    # under either name.
    for s in KNOWN_ANSWERS["b0_e1p_dense"]:
        d, m = KNOWN_ANSWERS["b0_e1p_dense"][s], KNOWN_ANSWERS["b0_e1p_moe48"][s]
        assert abs(d - m) > 0.02, (
            f"step {s}: the two comparators' gaps are {d:+.4f} and {m:+.4f}, within 0.02 -- "
            f"this world can no longer tell a relabelled curve from the right one")
    print("  ok   the two curves differ by >0.02 at every point, so a relabel is detectable")

    # PAIRING AT MATCHED STEPS INSTEAD OF MATCHED TOKENS. ratio=1 is the error the whole
    # exercise guards against, and against the dense arm it does not merely shift the numbers --
    # it REVERSES THE SIGN, reporting the MoE arm ahead at every point.
    same_step = {s: g for s, _t, _a, _c, g in pairs(arm, dense, 1)}
    assert all(g < 0 for g in same_step.values()), \
        f"matched-STEP pairing was expected to reverse the sign; got {same_step}"
    assert all(same_step[s] * KNOWN_ANSWERS["b0_e1p_dense"][s] < 0 for s in same_step), \
        "matched-step and matched-token gaps do not disagree in sign, so ratio=1 is undetectable"
    print("  ok   matched-STEP pairing reverses the sign, so the 3x ratio is load-bearing")

    # A comparator that stops early must shorten the table, not fabricate a pair.
    short = {k: v for k, v in dense.items() if k <= 1200}
    got = pairs(arm, short, 3)
    assert [p[0] for p in got] == [200, 400], \
        f"a comparator with points to step 1200 paired {[p[0] for p in got]}, expected [200, 400]"
    print("  ok   a comparator that stops early yields fewer pairs, never an invented one")

    # tok/step comes off the cfg line. The arm's own line must give 786,432 and the comparators'
    # 262,144: if a future launch changes batch, the 3x ratio is wrong and this catches it.
    cfg_arm = "cfg batch 8 accum 4 seq 4096 grad_ckpt False"
    cfg_cmp = "cfg batch 16 accum 2 seq 4096 grad_ckpt False"
    assert parse_tok_per_step(cfg_arm, 6) == 786432, parse_tok_per_step(cfg_arm, 6)
    assert parse_tok_per_step(cfg_cmp, 2) == 262144, parse_tok_per_step(cfg_cmp, 2)
    assert parse_tok_per_step("no cfg here", 6) is None
    print("  ok   tok/step is read from the cfg line: 786,432 vs 262,144, ratio 3")

    print("\nequal_token_gap selftest OK: both known-answer sets reproduce, the curves are "
          "distinguishable,\nmatched-step pairing is caught by a sign reversal, and the token "
          "ratio comes from the logs")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--cards-arm", type=int, default=6, help="cards the MoE arm ran on")
    ap.add_argument("--cards-cmp", type=int, default=2, help="cards the comparators ran on")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()

    arm_text = pod_read(ARM)
    arm_val = parse_val(arm_text)
    if not arm_val:
        print(f"REFUSING: no val lines in {ARM}.log", flush=True)
        return 1
    arm_tok = parse_tok_per_step(arm_text, a.cards_arm)
    if arm_tok != TOK_PER_STEP[ARM]:
        print(f"REFUSING: {ARM}'s cfg line gives {arm_tok} tok/step, not "
              f"{TOK_PER_STEP[ARM]} -- the 3x pairing no longer holds", flush=True)
        return 1

    rc = 0
    for name, label in COMPARATORS.items():
        text = pod_read(name)
        cmp_val = parse_val(text)
        cmp_tok = parse_tok_per_step(text, a.cards_cmp)
        if cmp_tok != TOK_PER_STEP[name]:
            print(f"REFUSING: {name}'s cfg line gives {cmp_tok} tok/step, not "
                  f"{TOK_PER_STEP[name]}", flush=True)
            rc = 1
            continue
        ratio = arm_tok // cmp_tok
        rows = pairs(arm_val, cmp_val, ratio)
        print(f"\n{ARM} vs {name} ({label}), ratio {ratio}x, {len(rows)} matched-token pairs:")
        for s, tok, av, cv, gap in rows:
            known = KNOWN_ANSWERS[name].get(s)
            mark = ""
            if known is not None:
                mark = "  KNOWN ok" if abs(gap - known) < 5e-4 else f"  KNOWN {known:+.3f} MISMATCH"
                if "MISMATCH" in mark:
                    rc = 1
            print(f"  step {s:5d}  {tok / 1e9:.2f}B tok   {av:.3f} - {cv:.3f} = {gap:+.4f}{mark}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
