#!/usr/bin/env bash
# AttnRes A/B at equal tokens: baseline vs --attn_res, same seed/data, N steps each.
#   NGPU=3 STEPS=500 scripts/run_ablation.sh [extra train.py args...]
# Compare: grep "^step" runs/ablation_base.log runs/ablation_attnres.log | tail
#
# BLOCKS defaults to 4 to match the standing recipe (`--attn_res --attn_res_blocks 4`).
# Passing --attn_res alone leaves Cfg.attn_res_blocks at 0, which means Full -- every sublayer a
# source. That is a different architecture from the one in production, so the arm would answer a
# question nobody asked. Set BLOCKS=0 deliberately to test Full.
#
# What else changes with the variable: the attnres arm adds the AttnRes pseudo-query parameters
# and its own AdamW group (attn_res_lr). Parameter count is not matched. Report throughput for
# both arms -- the measured tax is 0.55-0.88x and it is half the decision.
set -euo pipefail
cd "$(dirname "$0")/.."
NGPU=${NGPU:-3}; STEPS=${STEPS:-500}; PORT=${PORT:-29530}; BLOCKS=${BLOCKS:-4}
for v in base attnres; do
  extra=""; [ "$v" = attnres ] && extra="--attn_res --attn_res_blocks $BLOCKS"
  torchrun --nproc_per_node="$NGPU" --master_port="$PORT" train.py \
    --fp8 --max_steps "$STEPS" --name "ablation_$v" $extra "$@"
done
python3 - <<'PY'
import re
for v in ("base", "attnres"):
    xs = [float(m) for m in re.findall(r"loss ([\d.]+)", open(f"runs/ablation_{v}.log").read())]
    print(f"{v:8s} last10 mean loss {sum(xs[-10:]) / len(xs[-10:]):.4f}")
PY
