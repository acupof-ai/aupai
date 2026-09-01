#!/usr/bin/env python3
"""The monitor must not close a KILLED run as ok (de, 2026-09-01).

The 22B milestone was killed at 04:22 to yield the lane. Its ledger row reads

    ok | process exited | monitor: process gone

byte-identical to what a completed score writes. Nothing on the board could see that
the 22B reading did not exist -- `harness ledger`, `gaps` and `score_matrix_present`
all read the status field, and the status said success.

The monitor watches a pid. A pid that vanishes has either finished or been killed, and
watching cannot tell those apart -- so it guessed, and guessed ok. cmd_launch now wraps
the child so the shell records $? to runs/<name>.rc, making the verdict an artifact
rather than an inference.

Four exit paths, no GPU. The fourth is the one that matters most: no .rc at all, which
is what a killed process GROUP leaves behind, because the wrapper dies with the job.
That must read fail -- a run whose fate is unknown is not a success.

    python3 scripts/test_monitor_exit.py

The monitor under test is the REAL generated source: _arm_monitor is called with
subprocess.Popen intercepted, so what runs is the shipped template with its paths
substituted, not a re-implementation that would share this probe's assumptions.
"""

import json
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))


def monitor_source(name, pid, log):
    """The exact source _arm_monitor would have spawned."""
    import harness

    holder = {}
    real = subprocess.Popen

    def fake(cmd, **kw):
        holder["code"] = cmd[2]

        class P:
            pid = 1

        return P()

    subprocess.Popen = fake
    try:
        harness._arm_monitor(name, pid, log)
    finally:
        subprocess.Popen = real
    return holder["code"]


def verdict(rc_content):
    """Run the real monitor against a dead pid and a given .rc state; return its row."""
    d = tempfile.mkdtemp(prefix="monexit_")
    name = "probe_run"
    log = os.path.join(d, f"{name}.log")
    open(log, "w").close()

    p = subprocess.Popen([sys.executable, "-c", "pass"])  # a pid that is certainly dead
    p.wait()

    rc_file = os.path.join(d, f"{name}.rc")
    if rc_content is not None:
        with open(rc_file, "w") as f:
            f.write(rc_content)

    calls = os.path.join(d, "calls.jsonl")
    stub = os.path.join(d, "exp_stub.py")
    with open(stub, "w") as f:
        f.write(
            "import json, sys\n"
            "a = sys.argv[1:]\n"
            "row = {a[i][2:]: a[i + 1] for i in range(1, len(a) - 1, 2)}\n"
            f"open({json.dumps(calls)}, 'a').write(json.dumps(row) + '\\n')\n"
        )

    code = monitor_source(name, p.pid, log)
    # Three redirections on the generated source: the ledger settled() reads, the rc
    # file, and exp.py. exp_py is assigned inside a TUPLE unpack sharing a line with
    # pid/log/name, so it is replaced by value rather than by line -- an `^exp_py = `
    # anchor matches nothing and a bare `exp_py = "..."` pattern eats the tuple.
    code = re.sub(r"^exp_log = .*$", f"exp_log = {json.dumps(os.path.join(d, 'ledger.jsonl'))}",
                  code, flags=re.M)
    code = re.sub(r"^rc_file = .*$", f"rc_file = {json.dumps(rc_file)}", code, flags=re.M)
    real_exp = os.path.join(ROOT, "scripts", "exp.py")
    assert f'"{real_exp}"' in code, "exp.py path not found in the generated monitor"
    code = code.replace(f'"{real_exp}"', json.dumps(stub))
    code = code.replace("time.sleep(60)", "time.sleep(1)")

    mon = os.path.join(d, "mon.py")
    with open(mon, "w") as f:
        f.write(code)
    r = subprocess.run([sys.executable, mon], capture_output=True, text=True, timeout=120)
    if not os.path.exists(calls):
        return None, f"monitor wrote no exp.py call (stderr: {r.stderr.strip()[-200:]})"
    with open(calls, encoding="utf-8") as f:
        return json.loads(f.readline()), None


CASES = [
    ("0", "ok", "exit 0"),
    ("7", "fail", "exit 7"),
    ("137", "fail", "exit 137 (signal 9)"),
    (None, "fail", "vanished"),
]


def main():
    bad = []
    for rc_content, want_status, want_result in CASES:
        label = f"rc={rc_content!r}"
        row, err = verdict(rc_content)
        if err:
            bad.append(f"{label}: {err}")
            continue
        if row.get("status") != want_status:
            bad.append(f"{label}: status {row.get('status')!r}, expected {want_status!r} "
                       f"(result {row.get('result')!r})")
        elif row.get("result") != want_result:
            bad.append(f"{label}: result {row.get('result')!r}, expected {want_result!r}")

    if bad:
        print("FAIL: the monitor's verdict does not follow the exit code")
        for b in bad:
            print(f"  {b}")
        print("\nA killed run closing ok is the 22B milestone incident: the row is "
              "indistinguishable from a completed one.")
        return 1
    print(f"OK: {len(CASES)} exit paths -- clean closes ok; nonzero, killed and "
          f"vanished all close fail")
    return 0


if __name__ == "__main__":
    sys.exit(main())
