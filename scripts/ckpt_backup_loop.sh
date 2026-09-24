#!/bin/bash
# Host-side backup loop: /data00/aupai_work/aupai -> /data01/aupai_backup/$NAME.
# Copy after the size is stable 120s; temp name; sha256 both sides; rename; MANIFEST; never delete.
set -u
NAME=${NAME:-v41_ced_0923}
SRC=${SRC:-/data00/aupai_work/aupai}
DST=${DST:-/data01/aupai_backup/$NAME}
PAT=ckpt_$NAME.pt
LOG=$DST/backup_loop.log
MAN=$DST/MANIFEST
INTERVAL=${INTERVAL:-120}
MAX_SECS=${MAX_SECS:-$((60*3600))}
MAX_DEAD_SECS=${MAX_DEAD_SECS:-$((48*3600))}
DEAD_EXIT=${DEAD_EXIT:-15}
LOG_CAP=${LOG_CAP:-5000000}

log() { echo "$(date -u +%FT%TZ) $*"; }

cap_log() {
  [ -f "$LOG" ] || return 0
  [ "$(stat -c %s "$LOG")" -gt "$LOG_CAP" ] || return 0
  tail -c 1000000 "$LOG" > "$LOG.tmp" && mv "$LOG.tmp" "$LOG" || return 0
  exec >> "$LOG" 2>&1
}

already() { grep -q " $1 " "$MAN" 2>/dev/null; }
final_backed() { already "$PAT"; }

# decide <dead_probes> <secs_since_train_vanished>: 0 exit, 5 give up, 1 keep polling.
# MAX_DEAD_SECS bounds the wait for a final ckpt that a dead run will never write.
decide() {
  [ "$1" -ge "$DEAD_EXIT" ] || return 1
  final_backed && return 0
  [ "$2" -ge "$MAX_DEAD_SECS" ] && return 5
  return 1
}

if [ "${1:-}" = "--selftest" ]; then
  _f=0
  _st() { if [ "$2" = "$3" ]; then echo "ok   $1"; else echo "FAIL $1 (want $2, got $3)"; _f=1; fi; }
  d=$(mktemp -d) || exit 1
  MAN="$d/MANIFEST"; touch "$MAN"

  already "$PAT"; _st "empty MANIFEST, nothing backed" 1 $?
  echo "$(date -u +%FT%TZ) $PAT 1 abc" >> "$MAN"; already "$PAT"; _st "plain final ckpt line" 0 $?

  : > "$MAN"
  echo "$(date -u +%FT%TZ) $PAT.step19999 1 abc" >> "$MAN"
  final_backed; _st "a step file does not read as the final ckpt" 1 $?

  decide 0 0;  _st "train alive -> keep" 1 $?
  decide 14 0; _st "14 probes, final backed -> keep (DEAD_EXIT is 15)" 1 $?
  decide 15 0; _st "gone 15 probes, no final ckpt -> keep" 1 $?
  decide 15 "$MAX_DEAD_SECS"; _st "gone and never finished -> give up" 5 $?
  echo "$(date -u +%FT%TZ) $PAT 1 abc" >> "$MAN"
  decide 15 0; _st "gone, final ckpt backed -> exit" 0 $?

  rm -rf "$d"
  [ "$_f" -eq 0 ] && echo "selftest: all worlds pass" || echo "selftest: FAILURES above"
  exit "$_f"
fi

mkdir -p "$DST"
touch "$MAN"
exec >> "$LOG" 2>&1
log "backup loop start pid $$ max ${MAX_SECS}s dead-cap ${MAX_DEAD_SECS}s interval ${INTERVAL}s"
START=$(date +%s)

back_up() {
  local f="$1" base tmp ss ds sz t0
  base=$(basename "$f")
  sz=$(stat -c %s "$f") || return 1
  sleep 120
  [ "$sz" = "$(stat -c %s "$f" 2>/dev/null)" ] || { log "skip $base: size still changing"; return 1; }
  already "$base" && { log "skip $base: already in MANIFEST"; return 0; }
  tmp=$DST/.tmp.$base.$$
  t0=$(date +%s)
  ionice -c3 cp "$f" "$tmp" || { log "cp failed $base"; rm -f "$tmp"; return 1; }
  ss=$(sha256sum "$f" | cut -d' ' -f1)
  ds=$(sha256sum "$tmp" | cut -d' ' -f1)
  if [ "$ss" != "$ds" ]; then log "SHA MISMATCH $base src=$ss dst=$ds"; rm -f "$tmp"; return 1; fi
  mv "$tmp" "$DST/$base"
  echo "$(date -u +%FT%TZ) $base $sz $ss" >> "$MAN"
  log "backed $base size $sz cp+verify $(( $(date +%s) - t0 ))s"
}

dead=0
gone_since=0
while :; do
  now=$(date +%s)
  [ $((now-START)) -ge "$MAX_SECS" ] && { log "hit ${MAX_SECS}s cap, exit"; exit 0; }
  cap_log
  while IFS= read -r f; do
    [ -z "$f" ] && continue
    b=$(basename "$f")
    case "$b" in
      *.step[0-9]*|$PAT) ;;
      *) continue ;;
    esac
    already "$b" || back_up "$f" || true
  done < <(ls -1 "$SRC"/$PAT* 2>/dev/null)

  if pgrep -f 'train.py.*--fp8' >/dev/null; then
    dead=0; gone_since=0
  else
    dead=$((dead+1))
    [ "$gone_since" -eq 0 ] && gone_since=$now
    gone_secs=$((now - gone_since))
    if [ "$dead" -ge "$DEAD_EXIT" ]; then
      log "no train proc (${gone_secs}s, final_backed=$(final_backed && echo yes || echo no))"
    else
      log "no train proc ($dead/$DEAD_EXIT)"
    fi
    decide "$dead" "$gone_secs"
    rc=$?
    if [ "$rc" -eq 0 ]; then log "training gone and $PAT backed up; loop exit"; exit 0; fi
    if [ "$rc" -eq 5 ]; then
      log "GIVE UP: training gone ${gone_secs}s and $PAT never appeared; loop exit"
      exit 5
    fi
  fi
  sleep "$INTERVAL"
done
