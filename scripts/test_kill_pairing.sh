#!/bin/bash
# Correct liveness: a zombie is DEAD. os.kill(pid,0) succeeds on zombies, which is
# what made my first matrix report killpg as failing (harness.py:6418 documents this trap).
set -u
work=$(mktemp -d); trap 'rm -rf "$work"' EXIT
printf '%s\n' '#!/bin/bash' 'sleep 300' 'echo post rc=$?' > "$work/plain.sh"
printf '%s\n' '#!/bin/bash' 'trap "kill 0" TERM INT' 'sleep 300 &' 'wait $!' 'echo post rc=$?' > "$work/bgwait.sh"
chmod +x "$work"/*.sh
probe() {
  python3 - "$work/$1.sh" "$2" <<'PY'
import os, pathlib, signal, subprocess, sys, time
script, mode = sys.argv[1], sys.argv[2]
p = subprocess.Popen(["bash", script], start_new_session=True,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(2)
k = subprocess.run(["pgrep","-P",str(p.pid)],capture_output=True,text=True).stdout.split()
if not k: print("SETUP-FAIL"); sys.exit(2)
kid = int(k[0])
os.kill(p.pid, signal.SIGTERM) if mode=="kill" else os.killpg(os.getpgid(p.pid), signal.SIGTERM)
time.sleep(2)
def state(pid):
    try: st = pathlib.Path(f"/proc/{pid}/stat").read_text().split()
    except FileNotFoundError: return "gone"
    return "zombie" if st[2] == "Z" else f"RUNNING({st[2]})"
print(state(kid))
try: os.kill(kid, 9)
except OSError: pass
PY
}
f=0
for w in plain bgwait; do for m in kill killpg; do
  r=$(probe "$w" "$m"); ok=no; case "$r" in gone|zombie) ok=yes;; esac
  echo "wrapper=$w harness=$m -> child=$r  reaped_or_gone=$ok"
  [ "$m" = killpg ] && [ "$ok" != yes ] && f=1
done; done
[ $f = 0 ] && echo "PASS: killpg terminates the child under both wrapper forms"
exit $f
