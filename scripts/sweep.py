"""Classify long-lived container processes into sweep classes; kill only with --execute.

    python3 scripts/sweep.py                 # report, the default
    python3 scripts/sweep.py --execute       # kill (a)(b)(c) by exact PID
    python3 scripts/sweep.py --selftest      # known answers, no processes touched

WHAT THIS IS FOR. 307 orphaned watchers accumulated over ~24 h on the pod and were killed by
hand on 2026-09-04 (ff035f77, 314 processes across three classes). The classes recur; the hand
sweep does not scale and its evidence lives in a commit message.

WHAT IT DELIBERATELY DOES NOT DO.

  ZOMBIES ARE COUNTED, NEVER SWEPT. 3,975 of them against 37 live processes at last measurement,
  because PID 1 is `sleep infinity`: a builtin blocked in sleep that never calls wait(). A zombie
  is already dead, no signal does anything, and only the PARENT's wait() frees the slot -- so
  nothing inside the container in any namespace can clear them. The fix is a reaping PID 1
  (tini/dumb-init, or `bash -c 'while :; do wait -n; done'`) or a container restart, which kills
  the user's tileRL jobs, so it is the user's to schedule. Reported as a standing fact.

  GPU HOLDERS ARE NOT CLASSIFIED HERE. Class (e) is host-only, by measurement: inside the
  container there is exactly ONE distinct cgroup across every live process -- PID 1, ours, and
  the tileRL trainers we must never touch are byte-identical -- so a cgroup test has zero
  discriminating power, and container `nvidia-smi` reports HOST pids that resolve in no container
  /proc. Intersecting the two pid sets marks everything foreign. This tool never claims a process
  is foreign or ours by cgroup or by GPU; that check belongs to a host-side tool via `tn exec`,
  keyed on GPU UUID plus cmdline, which are the only cross-boundary identities.

  A CLASS WITH NO RECORDED INSTANCE GETS NO MATCHER. (a), (b) and (c) are built because
  ff035f77 recorded 306, 3 and 5 real instances. Nothing else is.

  UNCLASSIFIED IS NEVER KILLED. Two traps make this load-bearing rather than cautious:
  b0's own first two runs of the (a) test reported "1 shared" because the scan counted its own
  fd, and the pid differed between runs, which is what exposed it; and PR-11 classified 307
  processes as another project's on cwd=/sgl-workspace/sglang, which is the CONTAINER'S DEFAULT
  CWD that every command not doing its own `cd` inherits -- it carries no ownership information.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time

# restartable: the report writes nothing, and --execute's ledger row is appended per PID
# IMMEDIATELY after that PID's signal, before the next one is considered -- so an interrupt loses
# no record of what was already killed, and re-running classifies the survivors afresh. There is
# no accumulated state to lose and nothing is written per shard because there are no shards: the
# unit of work is one process. Re-running after an interrupt is not merely cheap, it is the
# intended recovery -- a killed process is simply no longer in the next scan's population, and
# the cmdline re-read before each signal means a re-run cannot act on a pid that has been reused.

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MIN_AGE_S = 3600

A, B, C, D, E, UNCLASSIFIED = "a-watcher", "b-waitloop", "c-stale-tail", "d-accounted", \
    "e-foreign", "unclassified"
SWEEPABLE = (A, B, C)

# SCOPE, from env_hygiene.md §2 "Scope, and the paths that are never in it". A process is a
# candidate only if its cwd is under /work/aupai; anything on /data00../data03 -- and in
# particular /data01/aupai/backup, the 365 GiB rsync that is the only copy surviving a pod
# restart -- is never in scope until a class names it. Enforced here as a positive test rather
# than a blacklist: a cwd this tool cannot place is unclassified, not sweepable.
IN_SCOPE_PREFIXES = ("/work/aupai",)
NEVER_IN_SCOPE = ("/data00", "/data01", "/data02", "/data03")

# KNOWN LIMIT, measured on the pod 2026-09-04 and left in place deliberately.
#
# The cwd scope test excludes the very shape that motivates class (b). Two loops, ages 61,976s
# and 61,720s, both `bash -lc until <cond> runs/<x>.log; do sleep N; done`, both with cwd
# /sgl-workspace/sglang -- and that cwd IS THE REASON THEY ARE STUCK: their `cd` never took, so
# `runs/...` resolves under the container default where no runs/ exists, while the real logs have
# said ALL-DONE since Sep 3 14:38 and FETCH_DONE since 14:27. They are ours, they are dead
# weight, and this tool reports them as out of scope.
#
# PR-11 made the mirror-image error -- it called 307 processes "not ours" from the same cwd -- and
# the correction (C1.5) is that the container's default cwd carries NO ownership information. It
# cannot prove foreignness and it cannot prove ours. My scope test then used it as a positive
# ownership signal from the safe side, which is the same unfounded inference with a harmless
# consequence instead of a destructive one.
#
# Not widened here. Scope is env_hygiene.md §2's ruling, not this file's, and the fix has to say
# what DOES place a process in /work/aupai when its cwd cannot: probably an fd or an argv path
# under /work/aupai, which is the positive evidence C1.5 used ("they hold FDs on our events
# file"). Raised for tilerl/6e rather than decided here, because widening a kill scope on my own
# reading is how a sweeper kills something it should not.


def _run(cmd):
    """Never raises. A missing binary returns empty output, which every caller reads as "this
    view cannot answer" -- the safe direction, since an exception here would make the sweeper
    unusable on any machine without nvidia-smi (the laptop, where the selftest runs).
    """
    try:
        return subprocess.run(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL)
    except (OSError, ValueError):
        return type("R", (), {"stdout": "", "stderr": "", "returncode": 127})()


def processes():
    """[{pid, ppid, age, stat, args}] for live processes; zombies excluded, counted separately."""
    r = _run(["ps", "-eo", "pid,ppid,etimes,stat,args"])
    out, zombies = [], 0
    for line in r.stdout.split("\n")[1:]:
        f = line.split(None, 4)
        if len(f) < 5:
            continue
        pid, ppid, age, stat, args = f
        if stat.startswith("Z"):
            zombies += 1
            continue
        try:
            out.append({"pid": int(pid), "ppid": int(ppid), "age": int(age),
                        "stat": stat, "args": args})
        except ValueError:
            continue
    return out, zombies


def _fd1(pid):
    try:
        return os.readlink(f"/proc/{pid}/fd/1")
    except OSError:
        return None


def _cwd(pid):
    try:
        return os.readlink(f"/proc/{pid}/cwd")
    except OSError:
        return None


def _cmdline(pid):
    """The raw cmdline, for the re-read immediately before a signal."""
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as fh:
            return fh.read()
    except OSError:
        return None


def _pipe_holders(inode, live_pids, me):
    """Which OTHER live pids hold this pipe inode on any fd.

    EXCLUDES the scanning process. b0's first two runs of this test reported "1 shared" and the
    pid differed between runs -- both times the other holder was the scanner itself. A false
    "shared" makes a real orphan look busy, so the exclusion is what makes the class usable.
    Re-checks liveness, because a dead holder's fd can still appear in the walk.
    """
    holders = []
    want = f"pipe:[{inode}]"
    for pid in live_pids:
        if pid == me:
            continue
        d = f"/proc/{pid}/fd"
        try:
            fds = os.listdir(d)
        except OSError:
            continue
        for fd in fds:
            try:
                if os.readlink(os.path.join(d, fd)) == want:
                    if os.path.isdir(f"/proc/{pid}"):
                        holders.append(pid)
                    break
            except OSError:
                continue
    return holders


_WAIT_RE = re.compile(r"(?:until|while)\s+(?:!\s*)?\[+\s*-([fed])\s+([^\s\]]+)\s*\]+")


def _closed_runs(root):
    """{run name: end} for experiments rows whose CURRENT row is closed.

    exp.fold, never the last line. A ledger is an event log: a correction is appended and the
    last row under (name, started) is what the row says now, so reading lines would report a
    superseded value as live. This exact mistake made a broken world vacuous the same day
    (de-53): the fixture keyed on rows[-1] and a close landed under that key.
    """
    p = os.path.join(root, "runs", "experiments.jsonl")
    if not os.path.exists(p):
        return {}
    sys.path.insert(0, os.path.join(root, "scripts"))
    try:
        import exp as _exp
    except ImportError:
        return {}
    rows = []
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    out = {}
    for r in _exp.fold(rows):
        if r.get("status") and r.get("status") != "running":
            out[r.get("name")] = r.get("ended")
    return out


def _gpu_pids():
    """Pids `nvidia-smi` reports as holding a GPU, as strings, from whatever view we are in.

    NOT an ownership test and never used as one -- env_hygiene.md §2(e) makes ownership a
    host-side question keyed on GPU UUID plus cmdline. Used here only as an EXCLUSION: "any GPU
    process" is never in scope, so a pid that appears here is unclassified whatever its shape.
    Inside the container these are host pids that resolve in no container /proc, which is exactly
    why this cannot decide ours-vs-foreign; but a pid that matches is still excluded, and a view
    that returns nothing excludes nothing rather than claiming the cards are free.
    """
    r = _run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"])
    return {x.strip() for x in r.stdout.split("\n") if x.strip().isdigit()}


def classify(procs, zombies, root=ROOT, now=None, me=None, gpu_pids=None):
    """[(proc, class, evidence)] -- one verdict per process, evidence always populated."""
    now = now or time.time()
    me = me or os.getpid()
    gpu_pids = _gpu_pids() if gpu_pids is None else gpu_pids
    live = [p["pid"] for p in procs]
    closed = _closed_runs(root)
    verdicts = []
    for p in procs:
        pid, args = p["pid"], p["args"]
        if p["age"] < MIN_AGE_S:
            verdicts.append((p, UNCLASSIFIED, f"younger than {MIN_AGE_S}s ({p['age']}s)"))
            continue
        if pid == 1:
            verdicts.append((p, UNCLASSIFIED, "PID 1"))
            continue
        if str(pid) in gpu_pids:
            verdicts.append((p, UNCLASSIFIED,
                             "nvidia-smi reports this pid holding a GPU -- never in scope, and "
                             "whose it is cannot be answered from here (env_hygiene §2(e))"))
            continue
        cwd = _cwd(pid)
        if cwd and cwd.startswith(NEVER_IN_SCOPE):
            verdicts.append((p, UNCLASSIFIED,
                             f"cwd {cwd} is on a never-in-scope filesystem (env_hygiene §2)"))
            continue
        if cwd and not cwd.startswith(IN_SCOPE_PREFIXES):
            verdicts.append((p, UNCLASSIFIED,
                             f"cwd {cwd} is outside {IN_SCOPE_PREFIXES[0]}; scope is a positive "
                             f"test, so a cwd this tool cannot place is never swept"))
            continue

        # (b) first: a wait loop's args are unambiguous and cheap to read.
        m = _WAIT_RE.search(args)
        if m:
            flag, target = m.group(1), m.group(2)
            cwd = cwd or root
            full = target if target.startswith("/") else os.path.join(cwd, target)
            exists = os.path.exists(full)
            parent = os.path.dirname(full) or "/"
            try:
                pdir_m = max([os.path.getmtime(os.path.join(parent, f))
                              for f in os.listdir(parent)] or [0])
            except OSError:
                pdir_m = 0
            started = now - p["age"]
            if not exists and pdir_m and pdir_m < started:
                verdicts.append((p, B, f"waits on -{flag} {full} which does not exist; its "
                                       f"directory's newest mtime {int(started - pdir_m)}s "
                                       f"predates the loop, so the producer is gone"))
            elif not exists:
                verdicts.append((p, UNCLASSIFIED,
                                 f"waits on {full}, absent, but {parent} moved since the loop "
                                 f"started -- a producer may still be running"))
            else:
                verdicts.append((p, UNCLASSIFIED,
                                 f"waits on {full}, which EXISTS -- the condition is satisfied "
                                 f"and the loop is stuck for another reason (cwd {cwd})"))
            continue

        fd1 = _fd1(pid)
        is_tail = bool(re.search(r"\b(tail|watch)\b", args))

        # (c) stale tail on a finished run's log, before (a): a lonely pipe and a finished run
        # can both hold, and (c) is the more specific statement.
        if is_tail:
            t = re.search(r"(?:^|\s)(\S*runs/([\w.\-]+)\.log)(?:\s|$)", args)
            if t:
                logpath, name = t.group(1), t.group(2)
                cwd = cwd or root
                full = logpath if logpath.startswith("/") else os.path.join(cwd, logpath)
                if name not in closed:
                    verdicts.append((p, UNCLASSIFIED,
                                     f"tails runs/{name}.log; no closed experiments row named "
                                     f"{name!r} (exact match only, no fuzzy)"))
                    continue
                try:
                    lm = os.path.getmtime(full)
                except OSError:
                    lm = None
                end = closed[name]
                end_s = None
                if isinstance(end, str):
                    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
                        try:
                            end_s = time.mktime(time.strptime(end, fmt))
                            break
                        except ValueError:
                            continue
                if lm is not None and end_s is not None and lm > end_s:
                    verdicts.append((p, UNCLASSIFIED,
                                     f"run {name} is closed but {full} is still growing "
                                     f"({int(lm - end_s)}s after the row's end) -- the ROW is "
                                     f"wrong, not the tail"))
                else:
                    verdicts.append((p, C, f"tails runs/{name}.log whose run closed "
                                           f"({end}); log not written since"))
                continue

        # (a) orphaned watcher: a1 lonely pipe, or a2 file-redirected watcher whose target run
        # is finished. a2 without (c) is not sufficient -- a watcher writing to a file for a LIVE
        # run is doing its job, and (c) above is the only thing that establishes "finished".
        if fd1 and fd1.startswith("pipe:["):
            inode = fd1[len("pipe:["):-1]
            holders = _pipe_holders(inode, live, me)
            if not holders:
                verdicts.append((p, A, f"stdout is {fd1} and no other live process holds that "
                                       f"inode -- its output reaches nobody"))
            else:
                verdicts.append((p, UNCLASSIFIED,
                                 f"stdout pipe {fd1} is read by {holders[:3]}"))
            continue
        verdicts.append((p, UNCLASSIFIED, f"no class establishes this process (stat {p['stat']}, "
                                          f"fd1 {fd1}, age {p['age']}s)"))
    return verdicts


def sweep(execute=False, root=ROOT, out=sys.stdout):
    procs, zombies = processes()
    verdicts = classify(procs, zombies, root=root)
    counts = {}
    for _p, cls, _e in verdicts:
        counts[cls] = counts.get(cls, 0) + 1
    print(f"sweep: {len(procs)} live process(es), {zombies} zombie(s)", file=out)
    if zombies:
        print(f"  ZOMBIES: {zombies}. NOT sweepable and not counted as candidates -- a zombie has "
              f"already exited, no signal reaches it, and only PID 1's wait() frees the slot. "
              f"PID 1 here never calls wait(). Fix is a reaping init (tini/dumb-init) or a "
              f"container restart, which stops the user's jobs: the user schedules it.", file=out)
    print(f"  selector: ps -eo pid,ppid,etimes,stat,args, age > {MIN_AGE_S}s, "
          f"zombies excluded", file=out)
    for cls in (A, B, C, UNCLASSIFIED):
        print(f"  {cls:14} {counts.get(cls, 0)}", file=out)
    rows = []
    for p, cls, why in verdicts:
        if cls in SWEEPABLE:
            print(f"  [{cls}] pid {p['pid']} age {p['age']}s: {p['args'][:90]}", file=out)
            print(f"      {why}", file=out)
            rows.append((p, cls, why))
    if not execute:
        if rows:
            print(f"\n{len(rows)} sweepable; --execute to kill them by exact PID", file=out)
        return 0
    ledger = os.path.join(root, "runs", "sweeper.jsonl")
    killed = skipped = 0
    with open(ledger, "a", encoding="utf-8") as fh:
        for p, cls, why in rows:
            pid = p["pid"]
            # RE-READ THE CMDLINE IMMEDIATELY BEFORE THE SIGNAL. A pid is reused and the scan is
            # minutes old by now. b0's hand sweep skipped 1 of 307 on exactly this check, and
            # that skip is the feature: pid 277143 was no longer the expected tail.
            before = _cmdline(pid)
            expect = p["args"].replace(" ", "")
            now_args = (before or b"").decode("utf-8", "replace").replace("\x00", "").replace(" ", "")
            if before is None or not now_args.startswith(expect[:40]):
                skipped += 1
                fh.write(json.dumps({"pid": pid, "cls": cls, "action": "skipped",
                                     "reason": "cmdline changed between scan and signal",
                                     "expected": p["args"][:120],
                                     "found": now_args[:120]}, ensure_ascii=False) + "\n")
                print(f"  SKIPPED pid {pid}: cmdline changed since the scan", file=out)
                continue
            try:
                os.kill(pid, 15)
            except OSError as e:
                skipped += 1
                fh.write(json.dumps({"pid": pid, "cls": cls, "action": "skipped",
                                     "reason": f"kill failed: {e}"}, ensure_ascii=False) + "\n")
                continue
            st = _run(["ps", "-o", "stat=", "-p", str(pid)]).stdout.strip()
            killed += 1
            fh.write(json.dumps({"pid": pid, "cls": cls, "action": "killed",
                                 "cmd": p["args"][:200], "evidence": why,
                                 "stat_after": st or "gone"}, ensure_ascii=False) + "\n")
            print(f"  killed pid {pid} ({cls}); ps stat after: {st or 'gone'}", file=out)
    print(f"\n{killed} killed, {skipped} skipped; rows in runs/sweeper.jsonl", file=out)
    return 0


def _selftest():
    """Known answers built from ff035f77's REAL populations, not invented shapes.

    Each case is a process shape that sweep actually saw on the pod, with the verdict the hand
    sweep reached. A fixture that cannot reproduce the recorded population certifies nothing.
    """
    import shutil
    import tempfile

    bad = []

    def case(ok, what):
        print(f"  {'ok  ' if ok else 'FAIL'} {what}")
        if not ok:
            bad.append(what)

    d = tempfile.mkdtemp(prefix="sweep_st_")
    try:
        os.makedirs(os.path.join(d, "runs"), exist_ok=True)
        os.makedirs(os.path.join(d, "scripts"), exist_ok=True)
        shutil.copy(os.path.join(HERE, "exp.py"), os.path.join(d, "scripts", "exp.py"))
        # A ledger where a run is CLOSED BY AN APPENDED ROW -- the fold case. Written in this
        # order on purpose: reading the last line per name would work, reading the FIRST would
        # not, and the point is that fold gives the current row.
        with open(os.path.join(d, "runs", "experiments.jsonl"), "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"name": "done_run", "started": "2026-09-03 10:00",
                                 "status": "running"}) + "\n")
            fh.write(json.dumps({"name": "done_run", "started": "2026-09-03 10:00",
                                 "status": "ok", "ended": "2026-09-03 11:00"}) + "\n")
            fh.write(json.dumps({"name": "live_run", "started": "2026-09-03 12:00",
                                 "status": "running"}) + "\n")
        closed = _closed_runs(d)
        case("done_run" in closed and "live_run" not in closed,
             f"_closed_runs folds an appended close and keeps a running row open: {closed}")

        old = time.time() - 400000  # ~4.6 d, C1.2's measured age
        log_closed = os.path.join(d, "runs", "done_run.log")
        open(log_closed, "w").write("x\n")
        os.utime(log_closed, (old, old))
        log_live = os.path.join(d, "runs", "live_run.log")
        open(log_live, "w").write("x\n")

        # C1.2's real shape: `until [ -f data/code_supply/measure_partial.json ]`, cwd
        # /work/aupai, 4.58-4.60 d, target absent, data/code_supply/ untouched since 08-30.
        supply = os.path.join(d, "data", "code_supply")
        os.makedirs(supply, exist_ok=True)
        stale = os.path.join(supply, "other.json")
        open(stale, "w").write("{}\n")
        os.utime(stale, (old - 1000, old - 1000))
        procs = [
            {"pid": 138628, "ppid": 1, "age": 396000, "stat": "S",
             "args": f"bash -c until [ -f {supply}/measure_partial.json ]; do sleep 5; done"},
            {"pid": 3874083, "ppid": 0, "age": 60821, "stat": "S",
             "args": f"bash -lc until [ -f {stale} ]; do sleep 5; done"},
            {"pid": 1238204, "ppid": 1, "age": 90000, "stat": "S",
             "args": f"tail -f {log_closed}"},
            {"pid": 1259883, "ppid": 1, "age": 90000, "stat": "S",
             "args": f"tail -f {log_live}"},
            {"pid": 296254, "ppid": 296253, "age": 6584, "stat": "Rl",
             "args": "python3 -u -c import sys; from tilerl.cli import main"},
            {"pid": 999001, "ppid": 1, "age": 100, "stat": "S",
             "args": f"tail -f {log_closed}"},
        ]
        v = {p["pid"]: (cls, why) for p, cls, why in classify(procs, 0, root=d, gpu_pids=set())}

        case(v[138628][0] == B,
             f"C1.2's real shape classifies (b): {v[138628][0]} -- {v[138628][1][:70]}")
        case(v[3874083][0] == UNCLASSIFIED and "EXISTS" in v[3874083][1],
             f"a loop whose target EXISTS is unclassified, not swept: {v[3874083][0]}")
        case(v[1238204][0] == C,
             f"C1.3's shape on a closed run classifies (c): {v[1238204][0]}")
        case(v[1259883][0] == UNCLASSIFIED,
             f"the SAME tail shape on a RUNNING run is unclassified: {v[1259883][0]} "
             f"-- {v[1259883][1][:60]}")
        case(v[296254][0] == UNCLASSIFIED,
             f"a tileRL trainer is never sweepable: {v[296254][0]}")
        case(v[999001][0] == UNCLASSIFIED and "younger" in v[999001][1],
             f"a young process is out of scope whatever its shape: {v[999001][1][:50]}")

        # SCOPE AND THE GPU EXCLUSION, both of which must beat a matching shape. env_hygiene §2
        # puts /data0* and any GPU process permanently out of scope, so a process that would
        # otherwise classify (b) must come back unclassified on either ground. The cwd cases can
        # only be exercised where /proc/<pid>/cwd exists, so they assert on the pid's own reading
        # rather than on a fabricated cwd -- a fixture that fakes /proc would test the fake.
        gpu_v = {p["pid"]: (cls, why) for p, cls, why
                 in classify(procs, 0, root=d, gpu_pids={"138628"})}
        case(gpu_v[138628][0] == UNCLASSIFIED and "GPU" in gpu_v[138628][1],
             f"a GPU-holding pid is unclassified even with a matching (b) shape: "
             f"{gpu_v[138628][0]} -- {gpu_v[138628][1][:60]}")
        case(gpu_v[138628][0] != B,
             "the GPU exclusion beats the class it would otherwise match")
        case(NEVER_IN_SCOPE == ("/data00", "/data01", "/data02", "/data03")
             and IN_SCOPE_PREFIXES == ("/work/aupai",),
             "scope constants match env_hygiene.md §2 verbatim (/work/aupai in, /data0* never)")
        # A cwd this tool cannot place must NOT sweep. Asserted through the real reader on this
        # very process, whose cwd is the repo and not /work/aupai -- so on a laptop the scope
        # test is what makes every verdict unclassified, which is the correct inert behaviour.
        own = _cwd(os.getpid())
        if own is not None:
            case(not own.startswith(IN_SCOPE_PREFIXES),
                 f"this checkout's cwd {own} is outside scope, so a laptop run sweeps nothing")

        # THE SCANNER MUST NOT COUNT ITS OWN FD. b0's first two runs reported "1 shared" for a
        # genuinely lonely pipe because of this, and the pid differed between runs.
        rd, wr = os.pipe()
        try:
            inode = os.readlink(f"/proc/{os.getpid()}/fd/{wr}") if os.path.exists(
                f"/proc/{os.getpid()}/fd/{wr}") else None
            if inode and inode.startswith("pipe:["):
                ino = inode[len("pipe:["):-1]
                holders = _pipe_holders(ino, [os.getpid()], os.getpid())
                case(holders == [],
                     "a pipe held ONLY by the scanner reads as lonely, not as shared")
            else:
                case(True, "no /proc/self/fd here (not Linux) -- pipe case skipped, not faked")
        finally:
            os.close(rd)
            os.close(wr)

        # ZOMBIES NEVER REACH classify(). The filter is in processes(), which is the only caller
        # that reads ps -- so the property to assert is that processes() counts a Z row and does
        # not return it. My first version of this case handed a fake Z row straight to classify()
        # and asserted the verdict list held no Z, which is trivially true of any list classify()
        # builds: it copies the dicts it is given. That assertion could not fail, i.e. it tested
        # nothing. Asserted against a real ps line instead.
        ps_out = ("    PID    PPID ELAPSED STAT COMMAND\n"
                  "      5       1  400000 Z    [bash] <defunct>\n"
                  " 138628       1  396000 S    bash -c until [ -f /nope ]; do sleep 5; done\n")
        real_run = globals()["_run"]
        globals()["_run"] = lambda cmd: type("R", (), {"stdout": ps_out, "returncode": 0})()
        try:
            got, z = processes()
        finally:
            globals()["_run"] = real_run
        case(z == 1 and [p["pid"] for p in got] == [138628],
             f"processes() counts the zombie and returns only the live row: z={z}, "
             f"pids={[p['pid'] for p in got]}")
    finally:
        shutil.rmtree(d, ignore_errors=True)

    print(f"sweep selftest: {'FAIL' if bad else 'ok'} ({len(bad)} failing)")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--execute", action="store_true",
                    help="kill classes a/b/c by exact PID; default is a report")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    return sweep(execute=a.execute)


if __name__ == "__main__":
    sys.exit(main())
