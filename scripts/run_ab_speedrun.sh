#!/usr/bin/env bash
# One speedrun A/B: baseline vs one flag, at equal tokens. ARM picks which flag.
#   CUDA_VISIBLE_DEVICES=3,4,5,6 ARM=zeroinit scripts/run_ab_speedrun.sh   # A/B (3)
#   CUDA_VISIBLE_DEVICES=3,4,5,6 ARM=shapelr  scripts/run_ab_speedrun.sh   # A/B (2a)
#   CUDA_VISIBLE_DEVICES=3,4,5,6 ARM=valueembed scripts/run_ab_speedrun.sh # A/B (4)
#
# NOTE for A/B (4) only: that arm is NOT parameter-matched (+33.6M, +16.3%). Every other arm
# here is. The exp row says so and the reading must not be quoted as a mechanism result.
#
# Two arms, equal tokens, equal seed, equal data, same process shape. The ONLY difference is
# --zero_init_out, and the parameter count is IDENTICAL between arms -- unlike run_ablation.sh's
# AttnRes A/B, which adds pseudo-query parameters. So this one really is a single-variable test
# and its delta needs no capacity caveat.
#
# WHY A SEPARATE FILE and not a flag on run_ablation.sh: that script's arms are named
# ablation_base/ablation_attnres and its comparison greps those two logs. Threading a second
# variable through it would make the arm names lie about which variable moved.
#
# STEP 0 LOOKS BROKEN AND IS NOT. In the zero arm, w13/qkv gradients are EXACTLY zero on the
# first step (the sublayer output is 0 * x), then recover on step 1. Do NOT kill the run for
# that -- scripts/test_zero_init_out.py pins the contract. What WOULD be fatal is still-zero
# upstream gradients after step 1, and that test is what tells the two apart.
#
# The reading is domain_loss + minimal_pairs against NOISE_THRESHOLDS (eval/score_matrix.py:78):
# minimal_pairs needs 11.5pt to be readable, domain_loss 0.24 nat. A 500-step delta smaller
# than that is NOT a result in either direction.
set -euo pipefail
cd "$(dirname "$0")/.."
STEPS=${STEPS:-500}
PORT=${PORT:-29542}
MIX=${MIX:-data/mix_200m_4b.json}

# CARDS is INHERITED, never written. CUDA_VISIBLE_DEVICES is not additive -- assigning it in a
# child REPLACES the caller's restriction instead of indexing into it, so a script that writes
# a physical index escapes whatever lane it was confined to (2f97e4a: a lane-card launch landed
# on GPU 0 and blocked a training block). So the caller sets the lane:
#
#   CUDA_VISIBLE_DEVICES=3,4,5,6 scripts/run_ab_zero_init.sh
#
# and this script only READS it, both for torchrun's process count and for the claim.
CARDS=${CUDA_VISIBLE_DEVICES:-}
if [ -z "$CARDS" ]; then
  # No literal example of the assignment here, deliberately: harness.py:6286 records that
  # check_device_set_honoured matches the TEXT, so the assignment form inside an echo reads
  # as a real assignment and fails the gate. That false positive is on purpose (parsing shell
  # quoting costs more surface than it saves), and mangling the example to slip past a grep
  # would leave a human reading a wrong command. The usage line at the top of this file is
  # the copyable form, and it lives in a comment, which the check skips.
  echo "refusing: this A/B needs its card lane set by the caller, not chosen here." >&2
  echo "See the usage line at the top of $0 for the exact form." >&2
  echo "Taking the whole machine by default is how an A/B lands on a production run." >&2
  exit 1
fi
NGPU=$(awk -F, '{print NF}' <<<"$CARDS")   # NF on the comma-split line; `seq -s,` miscounts

# ARM is the variable under test: zeroinit (A/B 3) or shapelr (A/B 2a). One launcher, because
# the two A/Bs differ only in which flag the second arm carries -- everything the comparison
# depends on (tokens, seed, data, process shape, parameter count) is identical either way.
ARM=${ARM:-zeroinit}
case "$ARM" in
  zeroinit)   ARM_FLAG="--zero_init_out" ;;
  shapelr)    ARM_FLAG="--muon_shape_lr" ;;
  valueembed) ARM_FLAG="--value_embed" ;;
  # b0-17 is TWO arms against one base, not one: --untie_head alone adds +33,619,968 params
  # (+16.3%) at the unchanged embed lr, isolating CAPACITY; adding --head_lr 0.003464 (nanochat
  # 0.004*(d/768)^-0.5 at d1024, against the embed group's 0.1) isolates the LR MECHANISM at
  # identical parameter count. Reporting only one of them would leave the other explanation open,
  # which is exactly what A/B (4) could not close.
  untiehead)     ARM_FLAG="--untie_head" ;;
  untieheadlr)   ARM_FLAG="--untie_head --head_lr 0.003464" ;;
  fp32logits)    ARM_FLAG="--attn_res_fp32_logits" ;;
  *) echo "refusing: ARM must be zeroinit, shapelr, valueembed, untiehead, untieheadlr or fp32logits, got '$ARM'" >&2; exit 1 ;;
esac

# SKIP_BASE reuses an existing base checkpoint instead of retraining it. Only valid when that
# checkpoint's world size, steps, mix and seed all match this launch -- a base at a different
# world has a different tokens/step, so the same step number is a different token position and
# the comparison silently changes its x-axis (this bit A/B (3) vs (2a): world 5 against world 4).
# The caller states which checkpoint it is reusing; nothing here can verify a claim about a file
# that another run produced.
SKIP_BASE=${SKIP_BASE:-0}

# THE CLAIM MUST NAME torchrun, NOT THIS SHELL. card_claim.py:283 (de-34) refuses a shell pid
# on sight, and acquiring before the job exists can only offer this script's own pid: PPID is
# the default holder and there is nothing else to name yet. A shell either execs away or
# outlives the job, so the card reads ORPHAN or stays held after training ended -- both
# happened 2026-09-03, in opposite directions, on this very launcher's arms.
#
# So each arm's torchrun is backgrounded FIRST and claimed by pid immediately, inside the
# sub-second window before CUDA init touches a card. The claim is per-arm, which also makes
# `card_claim.py status` name the arm that holds the lane instead of just the A/B.
#
# The guard's own hint does not get you here: with no torchrun yet, the only python descendant
# it can suggest is the `card_claim.py acquire` process asking the question.
CLAIM=""
trap '[ -n "$CLAIM" ] && python3 scripts/card_claim.py release --name "$CLAIM" >/dev/null 2>&1 || true' EXIT

ARMS="base $ARM"
[ "$SKIP_BASE" = "1" ] && ARMS="$ARM"
for v in $ARMS; do
  extra=""
  [ "$v" = "$ARM" ] && extra="$ARM_FLAG"
  echo "=== arm $v (cards $CARDS, $STEPS steps) $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  torchrun --nproc_per_node="$NGPU" --master_port="$PORT" train.py \
    --mix "$MIX" --max_steps "$STEPS" --name "ab_${ARM}_$v" \
    --dim 1024 --layers 12 --heads 8 --ffn_hidden 3072 --batch 16 --accum 2 \
    --no-grad_ckpt --lr_scale 1.0 --warmdown 0.1 --anneal_frac 0 --warmup 300 \
    --save_every "$STEPS" $extra "$@" &
  job=$!
  CLAIM="ab_${ARM}_$v"
  # WAIT FOR THE EXEC, don't just wait a bit. `cmd &` forks first and execs second, so for a
  # brief window the child's cmdline is still this bash -- and the de-34 guard reads exactly
  # that, so an immediate claim can be refused as "a shell" for a job that is in fact torchrun a
  # millisecond later.
  #
  # POLL argv[0], NOT A SUBSTRING. Pre-exec, this shell's cmdline already CONTAINS the words
  # "torchrun" and "train.py" -- they are its arguments -- so `case *torchrun*` matches the very
  # shell we are waiting to stop being, and the wait ends immediately having tested nothing.
  # That is card_claim.py:119's documented trap, reproduced here by writing the check the easy
  # way first. `ps -o args=` because the guard falls back to it too (no /proc on macOS, where
  # the selftests run).
  for _ in $(seq 1 100); do
    a0=$(ps -o args= -p "$job" 2>/dev/null | awk '{print $1}')
    case "$(basename "${a0:-sh}")" in
      sh|bash|dash|zsh|ksh|"") ;;                # still the pre-exec shell: keep waiting
      *) break ;;                                # execed into something that is not a shell
    esac
    kill -0 "$job" 2>/dev/null || break          # died on its own: let the claim report the truth
    sleep 0.1
  done
  if ! python3 scripts/card_claim.py acquire --name "$CLAIM" --cards "$CARDS" \
       --pid "$job" --note "A/B $ARM arm $v, $STEPS steps" --wait "${CLAIM_WAIT:-0}"; then
    # Kill by the exact pid we started, never a pattern: this tree is shared by six sessions.
    kill "$job" 2>/dev/null || true
    CLAIM=""
    echo "REFUSING: could not claim cards $CARDS for arm $v -- torchrun $job killed before it" >&2
    echo "reached CUDA init. See card_claim.py status for who holds the lane." >&2
    exit 1
  fi
  wait "$job"
  python3 scripts/card_claim.py release --name "$CLAIM" >/dev/null 2>&1 || true
  CLAIM=""
done

ARM="$ARM" SKIP_BASE="$SKIP_BASE" BASE_CKPT="${BASE_CKPT:-}" python3 - <<'PY'
import os
import re
out = {}
arm = os.environ.get("ARM", "zeroinit")
# With SKIP_BASE the base log belongs to the run that produced the reused checkpoint, under
# THAT run's arm name, so it is absent here by construction. Reading it would fire the
# name-disagreement refusal below -- a message that says "the run may have completed fine"
# and would be read as a failure of a launch that did exactly what it was told.
arms = (arm,) if os.environ.get("SKIP_BASE") == "1" else ("base", arm)
if len(arms) == 1:
    print(f"SKIP_BASE=1: no base arm in this launch. Train loss below is {arm} ALONE -- there "
          f"is no delta to print here. The comparison runs against "
          f"{os.environ.get('BASE_CKPT') or 'the reused base checkpoint'} in eval, and its "
          f"world/steps/mix/seed match is the caller's claim, recorded in the exp row.")
for v in arms:
    path = f"runs/ab_{arm}_{v}.log"
    try:
        xs = [float(m) for m in re.findall(r"loss ([\d.]+)", open(path).read())]
    except FileNotFoundError:
        # NOT a soft "no log": the writer above and this reader must agree on the name, and
        # they did not -- train.py was given --name ab_zi_$v while this looked for
        # ab_${ARM}_$v, so a completed A/B printed "no log" for both arms and no delta at all.
        # Found by reading the pod's runs/ directory, not by anything going red.
        print(f"REFUSING: {path} does not exist. The arm log name and this reader disagree; "
              f"the run may have completed fine -- check runs/ for what train.py actually "
              f"wrote before rerunning anything.")
        continue
    if not xs:
        print(f"{v}: log has no loss lines"); continue
    out[v] = sum(xs[-10:]) / len(xs[-10:])
    print(f"{v:9s} last10 mean loss {out[v]:.4f}  ({len(xs)} loss lines)")
if len(out) == 2:
    d = out[arm] - out["base"]
    print(f"\ndelta ({arm} - base) {d:+.4f} nat")
    # ds.seed_variance_0p2b: sd 0.0516 nat, readable move 0.24 nat.
    print("READABLE" if abs(d) >= 0.24 else
          f"NOT READABLE: |{d:+.4f}| < 0.24 nat (ds.seed_variance_0p2b). Train loss is a "
          f"smoke check only -- the decision is domain_loss + minimal_pairs on the ckpts.")
PY
