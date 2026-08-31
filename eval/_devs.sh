# Map N shards onto the caller's CUDA_VISIBLE_DEVICES. Sourced, not executed.
#
#   source eval/_devs.sh "$N"        # then use ${_DEVS[$i]} for shard i
#
# CUDA_VISIBLE_DEVICES is not additive: an assignment in a child REPLACES the
# parent's restriction rather than indexing into it. A script that writes a
# physical index escapes whatever lane the caller confined it to -- on 2026-08-31
# a lane-card launch (CUDA_VISIBLE_DEVICES=7) of eval/code_zh.py landed on GPU 0,
# a training-block card, and blocked t01 (2f97e4a).
#
# 2f97e4a's per-script fix was ${_DEVS[$i]:-$i}, which still spills whenever N
# exceeds the caller's device count: CUDA_VISIBLE_DEVICES=7 with the default N=6
# puts shard 0 on GPU 7 and shards 1-5 on physical 1-5. A shard with no device to
# land on is a caller error, so it refuses instead of falling back.
_devs_init() {
  local n=$1
  IFS=',' read -ra _DEVS <<< "${CUDA_VISIBLE_DEVICES:-}"
  if [ "${#_DEVS[@]}" -eq 0 ] || [ -z "${_DEVS[0]}" ]; then
    # Caller imposed no restriction: the physical first N, the pre-2f97e4a default.
    _DEVS=()
    for ((i = 0; i < n; i++)); do _DEVS+=("$i"); done
    return 0
  fi
  if [ "$n" -gt "${#_DEVS[@]}" ]; then
    echo "refusing: $n shards but CUDA_VISIBLE_DEVICES exposes ${#_DEVS[@]} device(s)" \
         "(${CUDA_VISIBLE_DEVICES}). Pass ngpu <= ${#_DEVS[@]}, or widen the device set." >&2
    return 1
  fi
  return 0
}
_devs_init "${1:?_devs.sh needs the shard count}" || exit 1
