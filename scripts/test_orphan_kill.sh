#!/bin/bash
set -u
work=$(mktemp -d); trap 'rm -rf "$work"' EXIT
mk() { printf '%s\n' "#!/bin/bash" "$2" 'sleep 300' 'rc=$?' 'echo post-child rc=$rc' 'exit $rc' > "$work/$1.sh"; chmod +x "$work/$1.sh"; }
mk bare ""
mk trap_fg 'set -m; trap "kill -- -$$" TERM INT'
# child in background + wait: bash can then run the trap, and the trap kills the group
mk trap_bg 'trap "kill 0" TERM INT'
sed -i.bak 's/^sleep 300$/sleep 300 \&\nwait $!/' "$work/trap_bg.sh"
run() {
  local v=$1 exp=$2
  setsid bash "$work/$v.sh" >"$work/$v.log" 2>&1 &
  sleep 2
  local w c; w=$(pgrep -f "bash $work/$v.sh" | head -1); c=$(pgrep -P "$w" | head -1)
  [ -n "$w" ] && [ -n "$c" ] || { echo "$v: SETUP FAILED"; return 1; }
  kill -TERM "$w"; sleep 2
  local ca; kill -0 "$c" 2>/dev/null && ca=ORPHANED || ca=dead
  kill -9 "$c" 2>/dev/null; pkill -9 -P "$w" 2>/dev/null
  echo "$v: child=$ca (expected $exp)"
  [ "$ca" = "$exp" ]
}
f=0
run bare ORPHANED || f=1
run trap_fg ORPHANED || f=1
run trap_bg dead || f=1
[ $f = 0 ] && echo "PASS: bare and foreground-trap both orphan; background child + wait + trap kills it"
exit $f
