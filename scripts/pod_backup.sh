#!/bin/bash
# Back up what a pod deletion would destroy. /work/aupai is a Kubernetes emptyDir: when the
# pod goes, every checkpoint, ledger and fact on it goes with it. /mnt/data02 is a real NVMe
# filesystem inside the container (3.5T, 2.8T free measured 2026-09-06), so the data is
# recoverable if somebody copies it there, and only then.
#
# THIS IS THE SECOND ACCEPTED FIX FOR root_durable, not a replacement for the first. Moving
# AUPAI_ROOT to the durable mount is still the stronger answer; nobody does it mid-campaign,
# which is why the check FAILed on every pod run and got --force'd along with the real reds.
# The check reads MANIFEST's mtime, so a backup nobody runs goes red on its own -- see
# check_root_durable and _durable_backup_state in scripts/harness.py.
#
# WHAT IS BACKED UP (4c's ruling, 2026-09-06):
#   - every ckpt_*.pt that is a run's FINAL or a milestone. NOT the rolling saves, which
#     train.py writes as `<ckpt_path>.step<N>` and `<ckpt_path>.interrupt.step<N>` -- suffixes
#     AFTER the .pt, so `ckpt_x.pt` is kept and `ckpt_x.pt.step400` is not. A run manifest
#     reproduces those; 75 finals are 84G, and the rolling set is unbounded.
#   - runs/*.jsonl (the ledgers), facts/, data/tokenizer.json, data/mix_*.json. 11M+1M+2.5M.
#   - NOT data/corpus or the token caches: rebuildable, and 2.8T does not hold them anyway.
#
# Runs ON THE POD (~/bin/pod "cd /work/aupai && bash scripts/pod_backup.sh"). It is not a
# push from the laptop: the files being copied only exist there.
set -euo pipefail

DEST_MOUNT="${AUPAI_BACKUP_DEST:-/mnt/data02}"
DEST="$DEST_MOUNT/aupai_backup"
SRC="${AUPAI_ROOT:-/work/aupai}"

# REFUSE IF THE DESTINATION IS NOT A SEPARATE FILESYSTEM. A directory that merely EXISTS at
# /mnt/data02 is the failure this repo has already paid for once: another session created
# /data00/aupai_raw, os.path.isdir went true, and a check began advising a move onto a path
# that was not a mount (see _is_mount, 2026-08-30). Backing up onto the same emptyDir we are
# protecting against would produce a MANIFEST -- and a green check -- for data that dies with
# the pod.
#
# NOT mountpoint(1). It is absent on the laptop where the selftest runs (/usr/bin/mountpoint
# exists on the pod only), so `mountpoint -q` failed for EVERY path here -- including `/` --
# and the selftest's refusal case passed because nothing could ever be accepted. Measured
# 2026-09-06 while writing it. Comparing st_dev against the parent is the same question and
# is answered by stat(1) on both platforms; -L so a symlinked mount resolves.
is_mounted() {
  local p=$1 parent
  [ -d "$p" ] || return 1
  # `/` is its own parent, so the st_dev comparison below says "same device" and calls the
  # root filesystem not-a-mount. Caught by the selftest's positive control, which exists
  # because the negative one alone is satisfied by a guard that refuses everything.
  [ "$p" = / ] && return 0
  parent=$(dirname "$p")
  [ "$(stat -Lf '%d' "$p" 2>/dev/null || stat -Lc '%d' "$p" 2>/dev/null)" \
    != "$(stat -Lf '%d' "$parent" 2>/dev/null || stat -Lc '%d' "$parent" 2>/dev/null)" ]
}

require_durable() {
  if [ ! -d "$DEST_MOUNT" ]; then
    echo "refusing: $DEST_MOUNT does not exist. Nothing was copied." >&2
    exit 1
  fi
  if ! is_mounted "$DEST_MOUNT"; then
    echo "refusing: $DEST_MOUNT is a directory, not a mounted filesystem." >&2
    echo "  A backup onto the same emptyDir dies with the pod it is protecting against," >&2
    echo "  while leaving a MANIFEST that reads as a backup. Nothing was copied." >&2
    exit 1
  fi
}

# The file list, one path per line, relative to SRC. Printed rather than globbed inline so
# --selftest can drive it against a fixture.
backup_paths() {
  local root=$1
  ( cd "$root" 2>/dev/null || exit 0
    # Finals and milestones: a .pt with nothing after it. `.step<N>` and `.interrupt.step<N>`
    # both land after the extension, so one anchored test excludes both.
    ls ckpt_*.pt 2>/dev/null || true
    ls runs/*.jsonl 2>/dev/null || true
    ls facts/*.json 2>/dev/null || true
    ls data/tokenizer.json 2>/dev/null || true
    ls data/mix_*.json 2>/dev/null || true
  ) | grep -v '\.pt\.' | sort -u
}

if [ "${1:-}" = "--selftest" ]; then
  d=$(mktemp -d); fails=0
  mkdir -p "$d/runs" "$d/facts" "$d/data"
  : > "$d/ckpt_run_a.pt"                     # a final -- kept
  : > "$d/ckpt_run_a.pt.step400"             # rolling -- excluded
  : > "$d/ckpt_run_a.pt.interrupt.step410"   # interrupt rolling -- excluded
  : > "$d/ckpt_b.milestone_keep_step900.pt"  # a milestone, .pt last -- kept
  : > "$d/runs/tasks.jsonl"; : > "$d/facts/efficiency.json"
  : > "$d/data/tokenizer.json"; : > "$d/data/mix_v3.json"
  mkdir -p "$d/data/corpus"; : > "$d/data/corpus/shard0.bin"   # rebuildable -- excluded
  got=$(backup_paths "$d" | tr '\n' ' ')
  for want in ckpt_run_a.pt ckpt_b.milestone_keep_step900.pt runs/tasks.jsonl \
              facts/efficiency.json data/tokenizer.json data/mix_v3.json; do
    case " $got " in *" $want "*) ;; *) echo "FAIL: $want missing from the list: $got" >&2; fails=1;; esac
  done
  # THE EXCLUSIONS ARE THE LOAD-BEARING HALF. Without them the list is "everything", which
  # passes every inclusion assertion above while copying an unbounded rolling set and 247GB
  # of rebuildable corpus.
  for never in ckpt_run_a.pt.step400 ckpt_run_a.pt.interrupt.step410 data/corpus/shard0.bin; do
    case " $got " in *" $never "*) echo "FAIL: $never must NOT be backed up: $got" >&2; fails=1;; esac
  done
  # The destination guard, driven rather than read, BOTH DIRECTIONS. The negative case alone
  # is satisfied by a guard that refuses everything -- which is exactly what the first version
  # did: it used mountpoint(1), which does not exist on this laptop, so `/` was refused too and
  # this assertion passed while the script could never run anywhere. Measured 2026-09-06.
  nodir="$d/not_a_mount"; mkdir -p "$nodir"
  if AUPAI_BACKUP_DEST="$nodir" bash "$0" --dry-run >/dev/null 2>&1; then
    echo "FAIL: a destination that is not a mountpoint was accepted" >&2; fails=1
  fi
  # `/` is a mount on every platform this runs on. It must get PAST require_durable and be
  # stopped by something else (the empty-source refusal), never by the durability guard.
  mkdir -p "$d/empty_src"
  out=$(AUPAI_BACKUP_DEST=/ AUPAI_ROOT="$d/empty_src" bash "$0" --dry-run 2>&1 || true)
  case "$out" in
    *"not a mounted filesystem"*)
      echo "FAIL: / was called 'not a mounted filesystem' -- the guard refuses every path, so" >&2
      echo "      the negative case above proves nothing: $out" >&2
      fails=1;;
  esac
  rm -rf "$d"
  [ "$fails" -eq 0 ] || { echo "pod_backup selftest: FAIL"; exit 1; }
  echo "pod_backup selftest ok: finals and milestones are listed, .step/.interrupt rolling saves and data/corpus are not, and a destination that is not a mounted filesystem is refused"
  exit 0
fi

require_durable

list=$(mktemp)
backup_paths "$SRC" > "$list"
n=$(grep -c . "$list" || true)
if [ "$n" -eq 0 ]; then
  echo "refusing: nothing to back up under $SRC -- an empty backup would still stamp a MANIFEST" >&2
  rm -f "$list"
  exit 1
fi

if [ "${1:-}" = "--dry-run" ]; then
  echo "would back up $n path(s) from $SRC to $DEST"
  rm -f "$list"
  exit 0
fi

mkdir -p "$DEST"
rsync -a --files-from="$list" "$SRC/" "$DEST/"

# MANIFEST LAST, and only after rsync exits 0 (set -e). It is what check_root_durable reads,
# so writing it before the copy would stamp a backup that did not finish -- the same ordering
# rule pod_push follows for the drift manifest.
#
# sha256 per file, not just sizes: a truncated copy has a plausible size and the point of the
# manifest is that somebody can tell whether the bytes are recoverable.
( cd "$DEST" && while IFS= read -r f; do
    [ -f "$f" ] || continue
    printf '%s  %s  %s\n' "$(sha256sum "$f" | cut -d' ' -f1)" "$(stat -c%s "$f")" "$f"
  done < "$list" ) > "$DEST/MANIFEST.tmp"
mv "$DEST/MANIFEST.tmp" "$DEST/MANIFEST"
rm -f "$list"

echo "pod_backup: $n path(s) -> $DEST, $(du -sh "$DEST" | cut -f1) total"
echo "MANIFEST written; scripts/harness.py check_root_durable reads its mtime (48h window)"
