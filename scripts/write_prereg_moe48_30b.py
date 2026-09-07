#!/usr/bin/env python3
"""Append the 1.5b-a0.2b-e48 30B run's prereg row to runs/prereg.jsonl.

A WRITER RATHER THAN A HAND-EDIT, so every number in the row is computed from the same formulas
train.py uses at launch instead of transcribed from a message. The step counts, the warmdown start,
the token totals and the two resume deadlines are all derived here; if a formula changes, this
refuses or produces the new number rather than shipping a stale one.

restartable: appends ONE line to runs/prereg.jsonl and nothing else. Refuses if the id already
exists, so a second run cannot double-append; an interrupt before the append leaves the ledger
untouched. No state to resume.
"""
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEDGER = os.path.join(ROOT, "runs", "prereg.jsonl")
ROW_ID = "moe48_30b_0907"

# THE LAUNCH SHAPE, from the grant on the pod (main 3dc79f16), not from a peer message.
BATCH, ACCUM, WORLD, SEQ = 8, 4, 6, 4096
PER_STEP = BATCH * ACCUM * WORLD * SEQ            # 786,432 tokens/step
RESUME_STEP = 9000
LAUNCH_TOTAL_TOKENS = 20_000_000_000              # the launch mix's total; 30B arrives at resume 1
RESUME1_TOTAL_TOKENS = 30_000_000_000
WARMDOWN = 0.1                                    # cfg.warmdown, from the 8B run's own cfg line
S_PER_STEP = 2.8228                               # facts/moe.json#moe.step_time_vs_dense_b192

# The 8B run this resumes, for the schedule-moved arithmetic.
PRIOR_TOTAL_STEPS = 10172


def plan_steps(total_tokens, resume_step):
    """Steps train.py will compute, by ITS arithmetic and not an approximation.

    train.py sets `total_steps = Cfg.epochs * (len(Xtr) // (Cfg.batch * Cfg.accum))` from the plan
    length, then `if getattr(Cfg, "_cursor_seeded", False) and resume_step: total_steps +=
    resume_step`. So the segment is the REMAINING tokens over the step size, floored, and the total
    is that plus the resume point -- NOT total_tokens // PER_STEP, which is what I first computed
    and which lands one step high (38,147 against 38,146 at 30B) because it ignores the resume
    split.
    """
    segment = (total_tokens - PER_STEP * resume_step) // PER_STEP
    return int(segment), int(segment) + resume_step


def warmdown_start(total):
    """train.py's warmdown_start: total - max(1, int(cfg.warmdown * total))."""
    return total - max(1, int(WARMDOWN * total))


def lr_mult(step, total, warmup=300):
    """train.py's lr_mult, copied so the WSD join value in this row is the one the run will print."""
    if step < warmup:
        return (step + 1) / warmup
    wd = warmdown_start(total)
    wd_steps = total - wd
    if step < wd:
        return 1.0
    progress = min(1.0, (step - wd) / wd_steps)
    return 0.5 * (1 + math.cos(math.pi * progress))


def main():
    with open(LEDGER, encoding="utf-8") as fh:
        rows = [json.loads(ln) for ln in fh if ln.strip()]
    if any(r.get("id") == ROW_ID for r in rows):
        sys.exit(f"REFUSING: {ROW_ID} already exists in runs/prereg.jsonl. This writer registers "
                 f"the row; amending one is a different operation and must not silently re-append.")

    launch_seg, launch_total = plan_steps(LAUNCH_TOTAL_TOKENS, RESUME_STEP)
    r1_seg, r1_total = plan_steps(RESUME1_TOTAL_TOKENS, RESUME_STEP)
    launch_wd = warmdown_start(launch_total)
    r1_wd = warmdown_start(r1_total)

    join = lr_mult(RESUME_STEP, launch_total)
    prior_wd = warmdown_start(PRIOR_TOTAL_STEPS)
    prior_lr_at_resume = lr_mult(RESUME_STEP, PRIOR_TOTAL_STEPS)
    prior_lr_at_9500 = lr_mult(9500, PRIOR_TOTAL_STEPS)

    # THE RESUME-1 DEADLINE IS THE WARMDOWN START, not an epoch ceiling, and the row must say which
    # binds. Two independent constraints:
    #   cot's 4-epoch ceiling at 4*supply/weight = 21.02B tokens
    #   the launch plan's warmdown start at 18.000B tokens
    # The warmdown one is EARLIER, so it binds; a resume after warmdown has begun re-anneals weights
    # that already annealed, which is the failure step9000 was chosen to avoid.
    cot_supply, cot_weight = 424_056_227, 0.0806945
    cot_ceiling_tokens = 4 * cot_supply / cot_weight
    wd_tokens = launch_wd * PER_STEP
    if wd_tokens >= cot_ceiling_tokens:
        sys.exit(f"REFUSING: the warmdown start ({wd_tokens / 1e9:.3f}B) is not earlier than cot's "
                 f"4-epoch ceiling ({cot_ceiling_tokens / 1e9:.3f}B), so this row's claim about "
                 f"which deadline binds is wrong and it must state the other one.")
    hours_to_wd = (launch_wd - RESUME_STEP) * S_PER_STEP / 3600

    row = {
        "id": ROW_ID,
        "registered": "2026-09-07T04:2xZ",
        "by": "b0",
        "charter": (
            f"runs/prereg.jsonl#{ROW_ID} (this row). Grant: runs/card_assignment.json note "
            f"2026-09-07T05:2xZ (main 3dc79f16, pod stamp 3dc79f16) -- launch_block_granted true, "
            f"block_cards 2,3,4,5,6,7. Written by scripts/write_prereg_moe48_30b.py, which derives "
            f"every step count from train.py's own formulas."
        ),
        "question": (
            "Does the 48-expert MoE arm, continued from the 8B checkpoint to 20B and then 30B "
            "tokens, separate from the batch-matched dense arm on the metrics that separated at 8B "
            "-- and does the HumanEval tie at 8B survive more tokens or break?"
        ),
        "registered_before": (
            f"THE RUN HAS NOT STARTED. Cards 2-7 read 0 MiB, no claim in card_claim status, no "
            f"process. The resume source is fixed and verified before any step: "
            f"ckpt_1.5b-a0.2b-e48_8b.pt.step9000, which holds `opt` (5 groups with populated "
            f"momentum buffers), row_cursor summing to 1,728,000 = 9000 x {BATCH} x {ACCUM} x "
            f"{WORLD} exactly, row_cursor_basis full_plan_prefix, row_cursor_seed 42, and "
            f"row_cursor_srcfp matching all nine live build_corpus_stats.json fingerprints. Zero "
            f"cursors will be discarded, so the save-time cursor identity stays CHECKED for the "
            f"whole run and any row_cursor_sum_unchecked field in a later checkpoint is a real "
            f"signal rather than expected noise."
        ),
        "why_now": (
            "facts/moe.json#moe.control_profile_vs_dense_b192 is the 8B endpoint: domain_bpb "
            "-0.027777 and lambada_en nll_per_byte -0.045415 favour MoE, while humaneval_bpb is a "
            "TIE whose two estimators disagree in sign (+0.003080 per-task, -0.000514 "
            "byte-weighted) over the same 164 tasks. Code is the objective and the measured "
            "deficit, which is also why the released mix weight was ruled onto the code role."
        ),
        "arm": {
            "name": "1.5b-a0.2b-e48_30b",
            "resume_from": "ckpt_1.5b-a0.2b-e48_8b.pt.step9000",
            "cards": "2,3,4,5,6,7 (world 6)",
            "config": (
                f"MoE-48, 1.48B total / ~0.2B active: 48 routed experts top_k 3 expert_ffn 768 "
                f"plus 1 shared, moe_layers 0-11, dim 1024 L12 h8 ffn3072 attn_every 4. batch "
                f"{BATCH} x accum {ACCUM} x world {WORLD} x seq {SEQ} = {PER_STEP:,} tokens/step. "
                f"fp8, grad_ckpt off -- the same launch shape the 8B run used, which measured "
                f"44.11-46.18 GiB peak per card, so ~49 GiB spare on H20."
            ),
            "launch_mix": (
                f"data/mix_1.5b-a0.2b-e48_20b_launch.json: the 8B run's composition at TOTAL="
                f"{LAUNCH_TOTAL_TOKENS / 1e9:.0f}B. NOT the 30B mix -- at 30B the 20B weights put "
                f"cot at 5.71 epochs, chatml 5.70, chat_qa 5.70, and the generator refuses them. "
                f"The 30B mix arrives at resume 1."
            ),
            "plan": (
                f"segment {launch_seg:,} steps + resume {RESUME_STEP:,} = total_steps "
                f"{launch_total:,}, warmdown from step {launch_wd:,}. Derived by train.py's own "
                f"arithmetic (total_steps from the plan length, then += resume_step when the plan "
                f"is cursor-seeded), not by "
                f"total_tokens // tokens_per_step -- that ignores the resume split and lands one "
                f"step high."
            ),
        },
        "schedule": (
            f"THE RESUME REJOINS AT FULL LR AND ANNEALS ONCE. step {RESUME_STEP} is pre-anneal "
            f"under the OLD schedule by {prior_wd - RESUME_STEP} steps (warmdown started at "
            f"{prior_wd:,} of {PRIOR_TOTAL_STEPS:,}), so its lr_mult there is "
            f"{prior_lr_at_resume:.4f}. Under the new total {launch_total:,} warmdown moves to "
            f"{launch_wd:,}, and the WSD JOIN prints lr_mult {join:.4f}. step9500 would NOT work: "
            f"its lr_mult under the old schedule is {prior_lr_at_9500:.4f}, already descending, so "
            f"resuming it would anneal twice. The step9000-vs-9500 choice is load-bearing.\n"
            f"train.py will print `SCHEDULE MOVED: this resume computes total_steps "
            f"{launch_total}, the checkpoint was written under {PRIOR_TOTAL_STEPS} "
            f"({launch_total - PRIOR_TOTAL_STEPS:+d})`. THAT WARNING IS EXPECTED AND CORRECT -- it "
            f"reports the plan being allocated for tokens already spent, which is what a "
            f"resume-to-a-larger-total is. It is loud and non-fatal by design -- a runlog call, "
            f"not a raise.\n"
            f"--max_steps MUST NOT BE PASSED. It caps total_steps via "
            f"`total_steps = min(total_steps, args.max_steps)` and is exempt from the warning via "
            f"`not args.max_steps` in the guard, on the reasoning that naming it IS asking for a "
            f"different total. "
            f"Passing --max_steps {PRIOR_TOTAL_STEPS} to silence the warning would hold the OLD "
            f"schedule and anneal at step {prior_wd:,}, {prior_wd - RESUME_STEP} steps into a "
            f"{launch_seg:,}-step segment."
        ),
        "routing_at_resume": (
            "MEASURED, from the arm's own logged diagnostics rather than a fresh fixture pass: "
            "runs/moe_diag.jsonl run=moe48_8b at step 9000 reads load_gini 0.1180, usage_frac "
            "1.0000, entropy_norm 0.9938. usage_frac is 1.0 on ALL 104 logged rows, so zero idle "
            "experts at every step, and the steady state (step>=2000, n=82) is gini 0.0842-0.1479 "
            "median 0.1216. This is recorded because a profiler fixture on a RANDOMLY INITIALISED "
            "router routed 4095 of 4096 tokens to one expert with 15/48 idle; that is a "
            "random-init property and the trained router does not have it -- a gini of 0.118 is "
            "arithmetically incompatible with it. NOTE the pooled range across all nine runs in "
            "moe_diag is 0.0208-0.6153 and the 0.6153 belongs to e1p_moe48, a DIFFERENT arm; "
            "quoting the pooled range as this arm's is the aggregate-hides-the-partition error."
        ),
        "readout_1_primary": (
            "The control panel at the 20B endpoint, against the same panel at 8B: domain_bpb "
            "unweighted mean, lambada_en acc and nll_per_byte, humaneval_bpb per-task and "
            "byte-weighted. Run as `eval/score_matrix.py --profile control --ngpu 1` on one card, "
            "the same invocation that produced the 8B pair, so the rows are comparable."
        ),
        "readout_1_bar": (
            "THE BAR IS FIXED HERE, BEFORE THE RUN, from the 8B pair. MoE at 8B: domain_bpb "
            "0.334243, lambada_en acc 0.288182 (ci95 half-width 0.012366), nll_per_byte 0.707467, "
            "humaneval_bpb 0.558973 per-task and 0.443569 byte-weighted. For 20B to be a "
            "CONTINUATION rather than a surprise, domain_bpb and nll_per_byte must not rise. For "
            "the HumanEval tie to BREAK, the two estimators must agree in sign and the gap must "
            "exceed the 0.003080 that the disagreeing pair spans at 8B -- a single estimator "
            "moving is not enough, because that is the state already observed."
        ),
        "readout_2_dense_pairing": (
            f"SECONDARY AND CONDITIONAL: there is no dense arm at 20B, and none is planned in this "
            f"row. Comparison to dense is therefore only available at the 8B point already "
            f"measured. If a dense 20B arm is later run, pairing is step-for-step at ratio 1 "
            f"because both sides are {PER_STEP:,} tokens/step -- but that pairing does NOT exist "
            f"yet and no claim in this row depends on it."
        ),
        "resumes_planned": (
            f"TWO RESUMES ARE PLANNED AND REGISTERED IN ADVANCE, so neither is a mid-run decision.\n"
            f"RESUME 1 (data): swap in the deduped code corpus as a NEW domain name plus the "
            f"fetched code, and move to the 30B mix -- plan becomes segment {r1_seg:,} + "
            f"{RESUME_STEP:,} = {r1_total:,} steps, warmdown from {r1_wd:,}. A NEW NAME, never a "
            f"rename: a new name has no cursor entry, takes neither discard branch "
            f"(train.py's `if row_cursor and name in row_cursor`), and contributes 0 to the "
            f"base via `int(counts[i]) + int(_base.get(n, 0))` -- which is correct, since it consumed 0 rows "
            f"before that segment, so the cursor identity keeps holding. A RENAME would trip the "
            f"fingerprint discard (the `corpus <want_fp> -> <live_fp>` branch) and restart that "
            f"domain at row 0. The deduped directory "
            f"must be a REAL directory with real shards: _assert_mix_domains refuses a "
            f"symlinked domain unconditionally, and allow_drift never pardons a symlink.\n"
            f"DEADLINE, and it is the warmdown start rather than an epoch ceiling: resume 1 must "
            f"land BEFORE step {launch_wd:,} = {wd_tokens / 1e9:.3f}B tokens, about "
            f"{hours_to_wd:.2f} h after launch at the measured {S_PER_STEP} s/step. A resume after "
            f"warmdown has begun re-anneals weights that already annealed, which is exactly what "
            f"choosing step9000 over step9500 avoided. cot's 4-epoch ceiling sits later at "
            f"{cot_ceiling_tokens / 1e9:.3f}B (4 x {cot_supply:,} / {cot_weight}), so the warmdown "
            f"deadline BINDS FIRST and the epoch ceiling is not the operative constraint.\n"
            f"RESUME 2 (speed): fold in the step-time fixes. THE GATE IS AT RESUME TIME, per the "
            f"user's 2026-09-07 order: s/step measured before and after on the same cards and the "
            f"same mix, and a fix that does not improve it does not go in. The baseline is this "
            f"run's own steady-state s/step, not the {S_PER_STEP} from the 8B run -- that figure "
            f"is a full-log median from a different window and its own entry reports a band "
            f"1.2974-1.3101 against dense, so using it as a before-value would compare two "
            f"windows rather than two builds."
        ),
        "stop_rules": {
            "nan_or_oom": (
                "Any NaN or OOM: stop, record the step, do not restart at a different batch. Peak "
                "at this shape and launch line measured 44.11-46.18 GiB on the 8B run, so an OOM "
                "means something changed and the change is the finding."
            ),
            "routing_collapse": (
                "usage_frac below 1.0 or load_gini above 0.30 sustained over three consecutive "
                "logged points: stop and report. The steady-state band is 0.0842-0.1479 over 82 "
                "points, so 0.30 is roughly double the observed maximum and is a threshold on the "
                "quantity that would actually fail, not on a proxy."
            ),
            "cursor_unchecked": (
                "A checkpoint carrying row_cursor_sum_unchecked: stop and read it. Nothing is "
                "discarded at launch (all nine srcfp verified), so that field appearing means a "
                "corpus changed mid-run or a domain's seed moved."
            ),
            "schedule_moved_twice": (
                f"A second SCHEDULE MOVED line with a delta this row does not predict: stop. The "
                f"expected one is {launch_total - PRIOR_TOTAL_STEPS:+d} at launch and one more at "
                f"resume 1; a third means a total nobody registered."
            ),
        },
        "instrument": (
            "train.py at the pod's stamped commit; eval/score_matrix.py --profile control for the "
            "endpoint panel; runs/moe_diag.jsonl for routing. The launch mix is generated ON THE "
            "POD by scripts/write_mix_500m.py, which refuses on a host lacking "
            "build_corpus_stats.json -- a Mac-generated mix once overwrote the pod's correct "
            "fingerprints, which is why that guard exists and why the mix is not built locally."
        ),
        "what_would_make_this_row_wrong": (
            f"(1) If the resume does not actually rejoin at lr_mult {join:.4f} -- read the WSD JOIN "
            f"line in the first 20 log lines and compare. (2) If tokens/step is not {PER_STEP:,}: "
            f"any different world size or accum breaks every token figure here and the "
            f"step-for-step comparability to the 8B curve. (3) If the launch mix's weights are not "
            f"the 8B run's composition -- the 20B build must be byte-comparable in weights to "
            f"mix_200m_8b.json's, since that file was produced by copying the same 20B weights. "
            f"(4) If resume 1 lands after step {launch_wd:,}, the double-anneal this row exists to "
            f"avoid has happened and the LR shape is not the registered one."
        ),
        "status": "registered",
    }

    with open(LEDGER, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"launch plan : segment {launch_seg:,} + {RESUME_STEP:,} = {launch_total:,} steps, "
          f"warmdown {launch_wd:,}")
    print(f"resume 1    : segment {r1_seg:,} + {RESUME_STEP:,} = {r1_total:,} steps, "
          f"warmdown {r1_wd:,}")
    print(f"WSD join    : lr_mult {join:.4f} at step {RESUME_STEP:,} "
          f"(old schedule: {prior_lr_at_resume:.4f} at 9000, {prior_lr_at_9500:.4f} at 9500)")
    print(f"SCHEDULE MOVED expected: {launch_total - PRIOR_TOTAL_STEPS:+d}")
    print(f"resume 1 deadline: step {launch_wd:,} = {wd_tokens / 1e9:.3f}B tokens "
          f"({hours_to_wd:.2f} h), before cot's ceiling at {cot_ceiling_tokens / 1e9:.3f}B")
    print(f"appended {ROW_ID} to runs/prereg.jsonl ({len(rows) + 1} rows)")


if __name__ == "__main__":
    main()
