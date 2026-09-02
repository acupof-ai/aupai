#!/usr/bin/env bash
# One speedrun A/B: baseline vs one flag, at equal tokens. ARM picks which flag.
#   CUDA_VISIBLE_DEVICES=3,4,5,6 ARM=zeroinit scripts/run_ab_speedrun.sh   # A/B (3)
#   CUDA_VISIBLE_DEVICES=3,4,5,6 ARM=shapelr  scripts/run_ab_speedrun.sh   # A/B (2a)
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
  zeroinit) ARM_FLAG="--zero_init_out" ;;
  shapelr)  ARM_FLAG="--muon_shape_lr" ;;
  *) echo "refusing: ARM must be zeroinit or shapelr, got '$ARM'" >&2; exit 1 ;;
esac

trap 'python3 scripts/card_claim.py release --name "ab_$ARM" || true' EXIT
python3 scripts/card_claim.py acquire --name "ab_$ARM" --cards "$CARDS"

for v in base "$ARM"; do
  extra=""
  [ "$v" = "$ARM" ] && extra="$ARM_FLAG"
  echo "=== arm $v (cards $CARDS, $STEPS steps) $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  torchrun --nproc_per_node="$NGPU" --master_port="$PORT" train.py \
    --mix "$MIX" --max_steps "$STEPS" --name "ab_zi_$v" \
    --dim 1024 --layers 12 --heads 8 --ffn_hidden 3072 --batch 16 --accum 2 \
    --no-grad_ckpt --lr_scale 1.0 --warmdown 0.1 --anneal_frac 0 --warmup 300 \
    --save_every "$STEPS" $extra "$@"
done

ARM="$ARM" python3 - <<'PY'
import os
import re
out = {}
arm = os.environ.get("ARM", "zeroinit")
for v in ("base", arm):
    try:
        xs = [float(m) for m in re.findall(r"loss ([\d.]+)", open(f"runs/ab_{arm}_{v}.log").read())]
    except FileNotFoundError:
        print(f"{v}: no log"); continue
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
