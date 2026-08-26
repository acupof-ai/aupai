#!/usr/bin/env bash
# AttnRes A/B at equal tokens: baseline vs --attn_res, same seed/data, N steps each.
#   NGPU=3 STEPS=500 scripts/run_ablation.sh [extra train.py args...]
# Compare: grep "^step" runs/ablation_base.log runs/ablation_attnres.log | tail
set -euo pipefail
cd "$(dirname "$0")/.."
NGPU=${NGPU:-3}; STEPS=${STEPS:-500}; PORT=${PORT:-29530}
for v in base attnres; do
  extra=""; [ "$v" = attnres ] && extra="--attn_res"
  torchrun --nproc_per_node="$NGPU" --master_port="$PORT" train.py \
    --fp8 --max_steps "$STEPS" --name "ablation_$v" $extra "$@"
done
python3 - <<'PY'
import re
for v in ("base", "attnres"):
    xs = [float(m) for m in re.findall(r"loss ([\d.]+)", open(f"runs/ablation_{v}.log").read())]
    print(f"{v:8s} last10 mean loss {sum(xs[-10:]) / len(xs[-10:]):.4f}")
PY
