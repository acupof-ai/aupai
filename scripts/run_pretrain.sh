#!/usr/bin/env bash
# One pretraining run end to end: experiment record -> torchrun -> record the result -> plot.
#
#   scripts/run_pretrain.sh <name> [train.py flags...]
#   NGPU=8 PORT=29600 HYP="what this run tests" scripts/run_pretrain.sh k4_attnres --fp8 --attn_res
#
# Refuses to start if the box already has GPU memory allocated: two runs that shared a port and a
# log file is how a previous session ended up with two RL jobs it could not tell apart. FORCE=1
# overrides, e.g. when the other user's reserved GPUs hold a context.
set -euo pipefail
cd "$(dirname "$0")/.."

NAME=$1
shift
NGPU=${NGPU:-8}
PORT=${PORT:-29600}
CMD="torchrun --nproc_per_node=$NGPU --master_port=$PORT train.py --name $NAME $*"

used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s}')
if [ "${FORCE:-0}" != "1" ] && [ "$used" -gt 2000 ]; then
  echo "run_pretrain: ${used}MiB already allocated on this box -- refusing to launch $NAME"
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv
  exit 1
fi
if ss -ltn 2>/dev/null | grep -q ":$PORT "; then
  echo "run_pretrain: port $PORT is already listening -- pick another PORT"
  exit 1
fi

python3 scripts/exp.py start --name "$NAME" --cmd "$CMD" \
  --notes "${NOTES:-$NGPU GPUs}" --hypothesis "${HYP:-}" >/dev/null || true

set +e
CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NGPU - 1))) torchrun --nproc_per_node="$NGPU" \
  --master_port="$PORT" train.py --name "$NAME" "$@"
RC=$?
set -e

TAIL=$(grep -E "^(ep |step )" "runs/$NAME.log" 2>/dev/null | tail -1 || true)
if [ $RC -ne 0 ]; then
  python3 scripts/exp.py done --name "$NAME" --status fail --result "train exited $RC | $TAIL" || true
  echo "$NAME FAILED rc=$RC"
  exit $RC
fi
python3 scripts/exp.py done --name "$NAME" --status ok --result "$TAIL" || true
python3 scripts/plot_curves.py "runs/$NAME.log" || true
echo "$NAME done: $TAIL"
