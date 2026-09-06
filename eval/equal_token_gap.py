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

# THE ARM'S LOG IS IN SEGMENTS AND THE FIRST ONE IS DEAD. b0_moe48_8b.log ends at step 1000
# with `SignalException: Process 1585157 got signal: 15` and b0_moe48_8b.rc holds 1; the run
# continues in 1.5b-a0.2b-e48_8b.log, which prints "Resumed from ckpt_b0_moe48_8b.pt.step1000
# (step 1000)" under a cfg line identical to the first segment's and the same total 10172.
# Reading only the first name is not a stale table, it is a table that DOES NOT SAY IT IS
# SHORT: measured 2026-09-06, that read returned exactly 5 pairs and printed "5 matched-token
# pairs" while steps 1200 and 1400 existed on the pod. Segments are read in order, last write
# wins, and their cfg tok/step must agree or the pairing is refused.
ARM_LOGS = ("b0_moe48_8b", "1.5b-a0.2b-e48_8b")

# name -> (label, what the confound list says)
COMPARATORS = {
    "b0_e1p_dense": "dense 0.2b",
    "b0_e1p_moe48": "MoE-48 at 0.2b batch",
    # THE RATIO-1 COMPARATOR (e1-46). Same 786,432 tok/step as the arm, so pairing is
    # step-for-step and no token arithmetic enters; and the same total 10,172, so both sides
    # enter their warmdown at step 9155 and NO pair is annealed-vs-stable. That is what the
    # batch-64 comparators cannot give: past their step 3434 one side is annealing, which caps
    # them at arm step 1144. Registered in runs/prereg.jsonl#matched_batch_dense_0906 before
    # the arm existed.
    "0.2b_8b_b192": "dense 0.2b at batch 192 (ratio 1)",
}

# tok/step, from the cfg line: batch x accum x cards x seq. Asserted against the log, never
# assumed -- a comparator launched at a different batch would silently break the 3N pairing.
TOK_PER_STEP = {ARM: 8 * 4 * 6 * 4096, "b0_e1p_dense": 16 * 2 * 2 * 4096,
                "b0_e1p_moe48": 16 * 2 * 2 * 4096,
                "0.2b_8b_b192": 8 * 4 * 6 * 4096}

# Cards each comparator ran on, for the tok/step assertion. The ratio-1 arm is world 6 like the
# arm itself, not world 2 like the batch-64 pair -- reading it with --cards-cmp 2 would compute
# 262,144 from a batch-192 cfg line and refuse for the wrong reason.
CMP_CARDS = {"b0_e1p_dense": 2, "b0_e1p_moe48": 2, "0.2b_8b_b192": 6}

# WSD warmdown fraction, identical in all three arms' cfg lines. train.py:3391 starts the
# warmdown at `total - max(1, int(warmdown * total))`, so the comparators (total 3815) enter it
# at 3434 and the arm (total 10172) at 9155 -- confirmed against the arm's own printed
# "warmdown starts at step 9155".
WARMDOWN_FRAC = 0.1

VAL_RE = re.compile(r"^step (\d+)/(\d+) val ([\d.]+)")
# train.py:3391 prints this on a WSD join. Reading it beats recomputing: the printed line is
# what the run's own scheduler used, and a recomputation silently disagrees if warmdown,
# total_steps or the formula ever differ from what this file assumes.
WARMDOWN_LINE_RE = re.compile(r"warmdown starts at step (\d+)")
VAL_EVERY_RE = re.compile(r"val_every (\d+)")


def warmdown_start(total):
    """First step of the cosine tail, by train.py:3391's formula.

    The FALLBACK only. Prefer warmdown_from_log: a fresh run prints no WSD-join line (only a
    resume does), so this is what covers the arm that starts from scratch, and the two must
    agree wherever both exist -- a selftest world asserts exactly that on the arm's real 9155.
    """
    return total - max(1, int(WARMDOWN_FRAC * total))


def warmdown_from_log(text, total=None):
    """The warmdown step this log's own scheduler used, or the computed one if it printed none.

    Returns (step, source) so a caller can say which it got. A log that prints the line AND
    disagrees with the formula is a real finding, not a rounding difference, so it raises:
    every pair's anneal flag depends on this number and quietly preferring one reading would
    make the flag describe a schedule nobody ran.
    """
    m = WARMDOWN_LINE_RE.search(text)
    computed = warmdown_start(total) if total else None
    if not m:
        if computed is None:
            raise RuntimeError("no 'warmdown starts at step N' line and no total to compute "
                               "one from -- the anneal phase cannot be located")
        return computed, "computed"
    printed = int(m.group(1))
    if computed is not None and printed != computed:
        raise RuntimeError(
            f"the log prints warmdown starting at step {printed} but total {total} at "
            f"warmdown {WARMDOWN_FRAC} computes {computed} -- the run used a schedule this "
            f"file does not model, and every anneal flag would describe the wrong one")
    return printed, "printed"


def parse_val_every(text):
    """val_every off the cfg line. None when absent."""
    m = VAL_EVERY_RE.search(text)
    return int(m.group(1)) if m else None

# The landed pairs, per comparator: {moe_step: gap}. Both sets are pinned because the mixup this
# script exists to prevent is exactly using one arm's numbers under the other's name.
KNOWN_ANSWERS = {
    "b0_e1p_dense": {200: 0.668, 400: 0.140, 600: 0.052, 800: 0.037, 1000: 0.038},
    "b0_e1p_moe48": {200: 0.740, 400: 0.175, 600: 0.082, 800: 0.067, 1000: 0.066},
}
KNOWN_ARM_VAL = {200: 3.465, 400: 2.676, 600: 2.465, 800: 2.384, 1000: 2.321}


def pod_read(name, missing_ok=False):
    """The val lines and the cfg line of <name>.log on the pod, as text.

    One pod call per log: ~/bin/pod re-parses its argument, so the command stays a plain grep
    with no parentheses or quoted prose (pod argv cannot carry prose).

    `missing_ok` returns "" for a log that does not exist yet. A COMPARATOR THAT HAS NOT
    LAUNCHED IS A NORMAL STATE, not an error: 0.2b_8b_b192 was registered before its arm
    existed, and without this the run crashed with a traceback AFTER printing two correct
    tables -- the reader gets a stack trace where the answer should be. A read that fails for
    any other reason still raises.

    THE GREP HAS TO FETCH EVERY LINE THE PARSERS READ, and it did not. warmdown_from_log looks
    for "warmdown starts at step N", which train.py prints on a WSD join; the grep asked only
    for val lines and the cfg line, so that line never arrived and the header printed
    "warmdown at 9155 (computed)" for an arm whose log states 9155 explicitly. The computed
    fallback agreed here, which is exactly why it went unnoticed -- a silent fallback that
    happens to be right is indistinguishable from a read that worked.
    """
    cmd = (f"grep -E '^step [0-9]+/[0-9]+ val ' {POD_RUNS}/{name}.log; "
           f"grep -m1 'cfg batch' {POD_RUNS}/{name}.log; "
           f"grep -m1 'warmdown starts at step' {POD_RUNS}/{name}.log")
    r = subprocess.run([POD, cmd], capture_output=True, text=True, timeout=300)
    if r.returncode != 0 and not r.stdout.strip():
        err = r.stderr or ""
        if missing_ok and "No such file or directory" in err:
            return ""
        raise RuntimeError(f"pod read of {name}.log failed: {err[:200]}")
    return r.stdout


def parse_val(text):
    """{step: val} from log text. Last write wins: a resumed run reprints steps it redoes."""
    out = {}
    for line in text.splitlines():
        m = VAL_RE.match(line.strip())
        if m:
            out[int(m.group(1))] = float(m.group(3))
    return out


def parse_total(text):
    """Total steps off any `step N/TOTAL` line, for the warmdown boundary. None if absent."""
    m = VAL_RE.search(text) or re.search(r"^step (\d+)/(\d+) ", text, re.M)
    return int(m.group(2)) if m else None


def read_arm():
    """(val, tok_per_step, total) for the arm, across every segment of its log.

    Segments in ARM_LOGS order, later writes winning, so a resumed step supersedes the one the
    dead segment printed. A segment whose cfg tok/step disagrees with the first is refused
    rather than merged: a resume at a different world size or accum breaks the pairing that the
    whole comparison rests on, and merging it would hide that behind a longer table.
    """
    val, tok, total, seen = {}, None, None, []
    for name in ARM_LOGS:
        text = pod_read(name)
        v = parse_val(text)
        t = parse_tok_per_step(text, 6)
        if not v and t is None:
            continue
        seen.append(name)
        if t is not None:
            if tok is not None and t != tok:
                raise RuntimeError(
                    f"{name}.log gives {t} tok/step but an earlier segment gave {tok} -- the "
                    f"arm changed batch across a resume and step-for-step pairing is void")
            tok = t
        total = parse_total(text) or total
        val.update(v)
    if not seen:
        raise RuntimeError(f"no segment of the arm's log was readable: {list(ARM_LOGS)}")
    return val, tok, total, seen


def parse_tok_per_step(text, cards):
    """tok/step read off the cfg line, so the pairing rests on the log and not on my arithmetic.

    Returns None when no cfg line is present (the selftest's fixtures carry only val lines).
    """
    m = re.search(r"cfg batch (\d+) accum (\d+) seq (\d+)", text)
    if not m:
        return None
    b, acc, seq = (int(x) for x in m.groups())
    return b * acc * cards * seq


def pairs(arm_val, cmp_val, ratio, cmp_total=None, cmp_wd=None, arm_wd=None):
    """[(moe_step, tokens, arm_val, cmp_val, gap, tail)] at matched tokens.

    Only steps where BOTH sides have a val point: the comparator stops at its own max step, and
    silently dropping the arm's later points would shorten the table without saying so.

    `tail` marks a pair where THE TWO SIDES ARE IN DIFFERENT LR PHASES -- one annealing, the
    other not. It is symmetric, because with a ratio-1 comparator either side can be the one in
    its tail, and an arm-only tail would go unflagged if the test only looked at the comparator.
    Measured 2026-09-06: the dense comparator's lr is 1.00e-02 through step 3400 and 5.41e-04 by
    3800, and its val drops 2.259 -> 2.128 across that span on the decaying lr rather than on
    tokens, while the arm at total 10172 stays at stable lr until 9155. Such pairs are returned
    and marked, never dropped: dropping them silently is the failure this script exists for.

    `cmp_wd`/`arm_wd` are the warmdown steps read from each log's own printed line where it has
    one (warmdown_from_log). `cmp_total` remains as the fallback for a comparator that printed
    none. For the ratio-1 arm both sides share total 10,172 and warmdown 9155, so no pair is
    mixed-phase by construction -- and the count of mixed pairs is therefore a CHECK on that
    construction, not a nuisance: a non-zero count means the totals diverged.
    """
    out = []
    wd_c = cmp_wd if cmp_wd is not None else (warmdown_start(cmp_total) if cmp_total else None)
    for s in sorted(arm_val):
        cs = s * ratio
        if cs not in cmp_val:
            continue
        cmp_annealing = bool(wd_c is not None and cs >= wd_c)
        arm_annealing = bool(arm_wd is not None and s >= arm_wd)
        out.append((s, s * TOK_PER_STEP[ARM], arm_val[s], cmp_val[cs],
                    round(arm_val[s] - cmp_val[cs], 4), cmp_annealing != arm_annealing))
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
        got = {s: g for s, _t, _a, _c, g, _w in pairs(arm, cmp_val, 3)}
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
    same_step = {s: g for s, _t, _a, _c, g, _w in pairs(arm, dense, 1)}
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

    # THE DEAD FIRST SEGMENT. b0_moe48_8b.log stops at step 1000 on signal 15 and the run
    # continues under a different name, so reading one name returns a table that is short
    # without saying so. Two worlds: the merge recovers the later points, and a segment that
    # resumed at a different batch is REFUSED rather than merged.
    seg1 = ("step 200/10172 val 3.465\nstep 1000/10172 val 2.321\n"
            "cfg batch 8 accum 4 seq 4096 grad_ckpt False\n")
    seg2 = ("step 1200/10172 val 2.280\nstep 1400/10172 val 2.241\n"
            "cfg batch 8 accum 4 seq 4096 grad_ckpt False\n")
    saved, texts = pod_read, {"seg1": seg1, "seg2": seg2}
    try:
        globals()["pod_read"] = lambda n: texts[n]
        globals()["ARM_LOGS"] = ("seg1", "seg2")
        val, tok, total, seen = read_arm()
        assert sorted(val) == [200, 1000, 1200, 1400], sorted(val)
        assert val[1400] == 2.241 and tok == 786432 and total == 10172, (val, tok, total)
        assert seen == ["seg1", "seg2"], seen
        only_first = parse_val(seg1)
        assert max(only_first) == 1000 and max(val) == 1400, (
            "the merge did not recover any point past the dead segment, so this world cannot "
            "tell the fix from the defect")
        print("  ok   segments merge: reading one name stops at 1000, both reach 1400")

        texts["seg2"] = seg2.replace("batch 8 accum 4", "batch 4 accum 4")
        try:
            read_arm()
            raise AssertionError("a resume at a different batch was merged instead of refused")
        except RuntimeError as e:
            assert "tok/step" in str(e) and "void" in str(e), str(e)
        print("  ok   a segment that resumed at fewer tokens/step is refused, not merged")

        # AND THE NEGATIVE CONTROL, which is what my first fixture got wrong: "batch 16 accum 2"
        # is 32 per rank exactly as "batch 8 accum 4" is, so it is the SAME 786,432 tok/step and
        # the pairing still holds. The guard compares tokens per step, not the batch spelling,
        # and a reshuffle at equal product must be ACCEPTED -- my first version asserted this
        # case refused, which would have made the guard reject a legitimate resume.
        texts["seg2"] = seg2.replace("batch 8 accum 4", "batch 16 accum 2")
        val2, tok2, _tot, _seen = read_arm()
        assert tok2 == 786432 and max(val2) == 1400, (tok2, sorted(val2))
        print("  ok   a reshuffle at equal tokens/step is accepted, not refused")
    finally:
        globals()["pod_read"] = saved
        globals()["ARM_LOGS"] = ("b0_moe48_8b", "1.5b-a0.2b-e48_8b")

    # THE COMPARATOR'S WARMDOWN. Its lr is 1.00e-02 through step 3400 and 5.41e-04 by 3800, and
    # its val falls 2.259 -> 2.128 across that span on lr alone. The arm at 10172 total is not
    # in its warmdown until 9155, so a pair past comparator step 3434 is annealed-vs-stable and
    # must be marked. Measured: the pair at arm step 1200 lands on comparator step 3600.
    assert warmdown_start(3815) == 3434, warmdown_start(3815)
    assert warmdown_start(10172) == 9155, warmdown_start(10172)
    late_arm = {1000: 2.321, 1200: 2.280, 1266: 2.260}
    late_dense = {3000: 2.283, 3600: 2.194, 3798: 2.130}
    marked = {s: w for s, _t, _a, _c, _g, w in pairs(late_arm, late_dense, 3, cmp_total=3815)}
    assert marked == {1000: False, 1200: True, 1266: True}, marked
    assert not any(w for *_r, w in pairs(late_arm, late_dense, 3)), \
        "the tail flag fired with no cmp_total, so it is not reading the comparator's total"
    print("  ok   pairs past comparator step 3434 are marked annealed-vs-stable")

    # tok/step comes off the cfg line. The arm's own line must give 786,432 and the batch-64
    # comparators' 262,144: if a future launch changes batch, the 3x ratio is wrong and this
    # catches it. THIS WORLD WAS DEAD FROM 06a47515 UNTIL e1-46: my edit for the two-segment fix
    # inserted its new worlds plus a `return 0` ABOVE this block, so the block sat after a
    # return and never ran -- and the selftest still printed OK, because a world that does not
    # execute cannot fail. The tell was in the output all along: no "tok/step is read from the
    # cfg line" line among the oks. Restored here, and the duplicated closing print that came
    # with the accident is gone.
    cfg_arm = "cfg batch 8 accum 4 seq 4096 grad_ckpt False"
    cfg_cmp = "cfg batch 16 accum 2 seq 4096 grad_ckpt False"
    assert parse_tok_per_step(cfg_arm, 6) == 786432, parse_tok_per_step(cfg_arm, 6)
    assert parse_tok_per_step(cfg_cmp, 2) == 262144, parse_tok_per_step(cfg_cmp, 2)
    assert parse_tok_per_step("no cfg here", 6) is None
    # ...and the ratio-1 comparator's own line, read at ITS card count. Reading a batch-192 cfg
    # line with the batch-64 pair's 2 cards computes 262,144 and would refuse for the wrong
    # reason, which is why CMP_CARDS exists rather than one --cards-cmp for every comparator.
    assert parse_tok_per_step(cfg_arm, CMP_CARDS["0.2b_8b_b192"]) == TOK_PER_STEP["0.2b_8b_b192"]
    assert parse_tok_per_step(cfg_arm, 2) != TOK_PER_STEP["0.2b_8b_b192"], \
        "the ratio-1 comparator's cfg line gives its right tok/step at the WRONG card count, " \
        "so CMP_CARDS is not load-bearing and this world proves nothing"
    print("  ok   tok/step is read from the cfg line at each comparator's own card count")

    n_new = _selftest_ratio1()

    print(f"\nequal_token_gap selftest OK: both known-answer sets reproduce, the curves are "
          f"distinguishable,\nmatched-step pairing is caught by a sign reversal, the arm's dead "
          f"segment is merged,\nmixed-phase pairs are marked, and the ratio-1 comparator's "
          f"{n_new} world(s) hold")
    return 0


def _selftest_ratio1():
    """The ratio-1 comparator (e1-46), on synthetic two-log fixtures.

    Returns the number of worlds asserted, so the closing line cannot claim worlds that were
    skipped -- the failure the block above this one just demonstrated.
    """
    ARM_CFG = "cfg batch 8 accum 4 seq 4096 val_every 200 warmup 300 warmdown 0.1"
    n = 0

    # THE WARMDOWN STEP COMES FROM THE LOG, NOT FROM MY ARITHMETIC. Both arms print total 10172
    # and warmdown 9155, so the printed and computed values must agree -- and where they do not,
    # that is a schedule this file does not model and the read must refuse rather than pick one.
    txt = ARM_CFG + "\nWSD JOIN: resumed at step 1000/10172 | warmdown starts at step 9155\n"
    assert warmdown_from_log(txt, 10172) == (9155, "printed")
    assert warmdown_from_log(ARM_CFG, 10172) == (9155, "computed"), \
        "a fresh run prints no WSD-join line, so the computed fallback has to cover it"
    try:
        warmdown_from_log(txt.replace("step 9155", "step 8000"), 10172)
        raise AssertionError("a printed warmdown disagreeing with the formula was accepted")
    except RuntimeError as e:
        assert "does not model" in str(e), str(e)
    n += 3
    print("  ok   warmdown is read from the log's printed line, computed only when absent, "
          "and a disagreement refuses")

    # NO PAIR IS MIXED-PHASE, BY CONSTRUCTION, and the count is the check on that. Both sides
    # share total 10,172 and warmdown 9155, so every pair is stable-vs-stable or tail-vs-tail --
    # including one at step 9200, inside BOTH warmdowns, which must NOT be flagged.
    arm = {200: 3.465, 1000: 2.321, 9200: 1.900, 10000: 1.850}
    cmp_ = {200: 2.900, 1000: 2.250, 9200: 1.880, 10000: 1.840}
    rows = pairs(arm, cmp_, 1, cmp_total=10172, cmp_wd=9155, arm_wd=9155)
    assert [r[0] for r in rows] == [200, 1000, 9200, 10000], [r[0] for r in rows]
    assert not any(r[5] for r in rows), \
        f"a mixed-phase flag fired on two arms that anneal together: {[(r[0], r[5]) for r in rows]}"
    assert rows[0][4] == round(3.465 - 2.900, 4), rows[0][4]
    assert rows[0][1] == 200 * 786432, rows[0][1]
    n += 4
    print("  ok   ratio 1: every val point pairs step-for-step and none is mixed-phase, "
          "including a pair inside both warmdowns")

    # THE FLAG IS SYMMETRIC, which the batch-64 world could not show: there the comparator was
    # always the one annealing. If the arm alone is in its tail, that pair is mixed too, and a
    # one-sided test would pass a comparison of an annealing arm against a stable comparator.
    only_arm = pairs({9200: 1.9}, {9200: 1.88}, 1, cmp_wd=None, cmp_total=None, arm_wd=9155)
    assert only_arm[0][5], "the arm being alone in its warmdown was not flagged"
    only_cmp = pairs({9200: 1.9}, {9200: 1.88}, 1, cmp_wd=9155, arm_wd=None)
    assert only_cmp[0][5], "the comparator being alone in its warmdown was not flagged"
    n += 2
    print("  ok   the mixed-phase flag is symmetric: either side alone in its tail is marked")

    # A val_every MISMATCH REFUSES. 100 against 200 would pair on the arm's grid and drop half
    # the comparator's points without saying so -- the exact shape of this script's own defect.
    assert parse_val_every(ARM_CFG) == 200
    assert parse_val_every(ARM_CFG.replace("val_every 200", "val_every 100")) == 100
    assert parse_val_every("no cfg") is None
    n += 3
    print("  ok   val_every is parsed from each cfg line, so a mismatch is detectable")

    # AND THE NEGATIVE WORLD 4c ASKED FOR: a comparator whose tok/step differs must refuse the
    # pairing rather than pair on a ratio it computed from the wrong batch. A batch-4 cfg line
    # at 6 cards is 393,216 -- exactly half -- so integer division would give ratio 2 and pair
    # arm step N against comparator step 2N, a table that looks fine and compares nothing.
    bad = parse_tok_per_step("cfg batch 4 accum 4 seq 4096 val_every 200", 6)
    assert bad == 393216, bad
    assert bad != TOK_PER_STEP["0.2b_8b_b192"], bad
    assert TOK_PER_STEP[ARM] // bad == 2, "the wrong-batch fixture must produce a plausible " \
        "integer ratio, or the refusal is not the thing being tested"
    n += 3
    print("  ok   a comparator at half the tok/step yields ratio 2, which the refusal catches")

    # AND THE FETCH ITSELF, which is where this went wrong twice over. warmdown_from_log can
    # only read a line pod_read actually greps for, and it did not: the command asked for val
    # lines and the cfg line only, so the header printed "(computed)" for an arm whose log
    # states 9155 explicitly. THE COMPUTED FALLBACK AGREED, which is why it survived a real run
    # -- a silent fallback that happens to be right looks identical to a read that worked. The
    # assertion is on the command string rather than on a parse, because the defect was in what
    # was requested, not in what was parsed.
    probe = []
    saved_run = subprocess.run
    try:
        subprocess.run = lambda argv, **kw: (probe.append(argv[1]),
                                             type("R", (), {"returncode": 0, "stdout": "",
                                                            "stderr": ""})())[1]
        pod_read("whatever")
    finally:
        subprocess.run = saved_run
    assert len(probe) == 1, probe
    for needle in ("val ", "cfg batch", "warmdown starts at step"):
        assert needle in probe[0], \
            f"pod_read does not fetch {needle!r}, so any parser reading it silently gets nothing"
    n += 3
    print("  ok   pod_read fetches every line the parsers read, warmdown line included")

    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--cards-arm", type=int, default=6, help="cards the MoE arm ran on")
    ap.add_argument("--only", help="one comparator name, instead of all of them")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()

    arm_val, arm_tok, arm_total, seen = read_arm()
    if not arm_val:
        print(f"REFUSING: no val lines in any segment of the arm's log: {list(ARM_LOGS)}",
              flush=True)
        return 1
    if arm_tok != TOK_PER_STEP[ARM]:
        print(f"REFUSING: {ARM}'s cfg line gives {arm_tok} tok/step, not "
              f"{TOK_PER_STEP[ARM]} -- the 3x pairing no longer holds", flush=True)
        return 1
    arm_text = pod_read(ARM_LOGS[-1])
    arm_ve = parse_val_every(arm_text)
    arm_wd, arm_wd_src = warmdown_from_log(arm_text, arm_total)
    print(f"{ARM}: {len(arm_val)} val points to step {max(arm_val)} of {arm_total}, "
          f"read from {len(seen)} segment(s): {', '.join(seen)}; val_every {arm_ve}, "
          f"warmdown at {arm_wd} ({arm_wd_src})")

    rc = 0
    todo = [a.only] if a.only else list(COMPARATORS)
    for name in todo:
        if name not in COMPARATORS:
            print(f"REFUSING: {name} is not a comparator: {list(COMPARATORS)}", flush=True)
            return 1
        label = COMPARATORS[name]
        text = pod_read(name, missing_ok=True)
        cmp_val = parse_val(text)
        if not text.strip():
            print(f"\n{ARM} vs {name} ({label}): NOT LAUNCHED -- no {name}.log on the pod. No "
                  f"pairs, no row. Registered in runs/prereg.jsonl before its arm existed.")
            continue
        if not cmp_val:
            print(f"\n{ARM} vs {name} ({label}): no val line yet -- NO PAIRS, and no row is "
                  f"written. The arm has to print a val point before it can be paired.")
            continue
        cmp_total = parse_total(text)
        cmp_tok = parse_tok_per_step(text, CMP_CARDS[name])
        if cmp_tok != TOK_PER_STEP[name]:
            print(f"REFUSING: {name}'s cfg line gives {cmp_tok} tok/step at "
                  f"{CMP_CARDS[name]} cards, not {TOK_PER_STEP[name]} -- the pairing this "
                  f"comparator was registered for does not hold", flush=True)
            rc = 1
            continue
        # A val_every MISMATCH IS A REFUSAL, not a warning. The two sides only pair where both
        # printed a val point, so a comparator at val_every 100 against the arm's 200 would
        # silently pair on the arm's grid and quietly drop half the comparator's points -- a
        # shorter table that does not say it is short, which is this script's own history.
        cmp_ve = parse_val_every(text)
        if arm_ve is not None and cmp_ve is not None and cmp_ve != arm_ve:
            print(f"REFUSING: {name} runs val_every {cmp_ve} against the arm's {arm_ve} -- the "
                  f"two val grids do not line up and pairing would drop points without saying "
                  f"so", flush=True)
            rc = 1
            continue
        cmp_wd, cmp_wd_src = warmdown_from_log(text, cmp_total)
        ratio = arm_tok // cmp_tok
        rows = pairs(arm_val, cmp_val, ratio, cmp_total=cmp_total, cmp_wd=cmp_wd, arm_wd=arm_wd)
        n_tail = sum(1 for r in rows if r[5])
        head = (f"\n{ARM} vs {name} ({label}), ratio {ratio}x, {len(rows)} matched-token pairs; "
                f"warmdown {cmp_wd} ({cmp_wd_src}) vs the arm's {arm_wd}")
        head += (f" -- {n_tail} pair(s) MIXED-PHASE" if n_tail
                 else " -- no mixed-phase pair" if ratio == 1 else "")
        print(head)
        for s, tok, av, cv, gap, tail in rows:
            known = KNOWN_ANSWERS.get(name, {}).get(s)
            mark = ""
            if known is not None:
                mark = "  KNOWN ok" if abs(gap - known) < 5e-4 else f"  KNOWN {known:+.3f} MISMATCH"
                if "MISMATCH" in mark:
                    rc = 1
            if tail:
                mark += ("  MIXED-PHASE: one side is in its lr warmdown and the other is not, "
                         "so this is not a token-matched comparison")
            print(f"  step {s:5d}  {tok / 1e9:.2f}B tok   {av:.3f} - {cv:.3f} = {gap:+.4f}{mark}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
