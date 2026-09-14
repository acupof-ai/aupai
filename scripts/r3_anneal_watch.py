#!/usr/bin/env python3
"""r3 anneal watcher: crash-class signals from the strings train.py actually prints.

train.py never renders a NaN loss line: the non-finite-grad branch in the
training loop hits `continue`, so the ONLY real signals are the two runlog
literals (see runlog calls beside the `n_skip` counter / good_state rollback):
- "step S/38070 non-finite grad — step skipped (n)"  (self-healed skip; WARN)
- "step S/38070  20 skips in a row — rolled back to snapshot"  (severe; CRASH)
Ten or more skips inside any rolling 500-step window escalate to CRASH even
without a rollback (silent degradation guard).

Other alerts:
- lowercase "nan" loss line: unconditional CRASH fallback.
- anneal-head loss jump: adjacent SAME-PHASE ([main]/[anneal]) lines inside the
  head window differing by > 0.5. Cross-phase pairs are different domain batches
  and swing ~1.0 by construction (false pages at step 34300/34310, 2026-09-14).
- gnorm blow-up: latest head-window gnorm > 5x the rolling window median.
- anneal val worsening: two consecutive anneal val points both above
  running-min + VAL_EPS (float jitter guard).

Never touches the training process. `--selftest` runs synthetic lines through
handle_line; no pod or log access.
"""
import os
import re
import statistics
import sys
import time

LOG = "/work/aupai/runs/v41_r3_0914.log"
OUT = "/work/aupai/runs/r3_anneal_watch.log"
ANNEAL = 34264
HEAD_END = ANNEAL + 200
JUMP = 0.5
GNORM_X = 5.0
GNORM_MIN_N = 5
VAL_EPS = 0.005
SKIP_WINDOW = 500
SKIP_CRASH_N = 10

LINE_RE = re.compile(r"step (\d+)/38070.*?(?:\d+% )?\[(main|anneal)\].*?\| loss ([0-9.]+|nan).*?gnorm ([0-9.]+)")
VAL_RE = re.compile(r"step (\d+)/38070 val ([0-9.]+)")
SKIP_RE = re.compile(r"step (\d+)/\d+ non-finite grad — step skipped \((\d+)\)")
ROLLBACK_RE = re.compile(r"step (\d+)/\d+ 20 skips in a row — rolled back to snapshot")


def new_state():
    return {
        "prev_step": 0, "prev_phase": None, "prev_loss": None,
        "gnorms": [], "val_seen": set(), "ann_min": None, "last_above": False,
        "skip_steps": [], "skip_escalated": False,
        "alerts": [],
    }


def handle_line(s, line):
    # train.py's real non-finite path, matched on literals before the periodic regex.
    rb = ROLLBACK_RE.search(line)
    if rb:
        s["skip_steps"] = []  # trainer reset its counter too
        s["skip_escalated"] = False
        s["alerts"].append(f"CRASH rollback to snapshot at step {rb.group(1)}: {line.strip()[:200]}")
        return
    sk = SKIP_RE.search(line)
    if sk:
        step = int(sk.group(1))
        s["skip_steps"] = [x for x in s["skip_steps"] if x > step - SKIP_WINDOW]
        s["skip_steps"].append(step)
        s["alerts"].append(f"WARN non-finite grad skip at step {step} "
                           f"({len(s['skip_steps'])} within {SKIP_WINDOW} steps)")
        if len(s["skip_steps"]) >= SKIP_CRASH_N and not s["skip_escalated"]:
            s["skip_escalated"] = True
            s["alerts"].append(f"CRASH {len(s['skip_steps'])} non-finite skips within "
                               f"{SKIP_WINDOW} steps at step {step}")
        if len(s["skip_steps"]) < SKIP_CRASH_N:
            s["skip_escalated"] = False
        return
    if "nan" in line.lower() and "step" in line:
        s["alerts"].append(f"CRASH NaN in log: {line.strip()[:200]}")
    m = LINE_RE.search(line)
    if m:
        step, phase, loss_s, g = int(m.group(1)), m.group(2), m.group(3), float(m.group(4))
        if ANNEAL <= step <= HEAD_END:
            s["gnorms"].append(g)
            if (s["prev_phase"] == phase and s["prev_loss"] is not None
                    and step != s["prev_step"] and loss_s != "nan"
                    and abs(float(loss_s) - s["prev_loss"]) > JUMP
                    and ANNEAL <= s["prev_step"] <= HEAD_END):
                s["alerts"].append(
                    f"anneal-head loss jump step={step} phase={phase} "
                    f"{s['prev_loss']}->{loss_s} d={float(loss_s) - s['prev_loss']:+.3f}")
            n = len(s["gnorms"])
            med = statistics.median(s["gnorms"][:-1]) if n >= 2 else 0.0
            if n >= GNORM_MIN_N and med > 0 and g > GNORM_X * med:
                s["alerts"].append(
                    f"CRASH gnorm blow-up step={step} gnorm={g:.3f} median={med:.3f}")
            s["prev_step"], s["prev_phase"], s["prev_loss"] = step, phase, float(loss_s)
    v = VAL_RE.search(line)
    if v:
        step, val = int(v.group(1)), float(v.group(2))
        if step >= ANNEAL and step not in s["val_seen"]:
            s["val_seen"].add(step)
            if s["ann_min"] is None:
                s["ann_min"] = val
            above = val > s["ann_min"] + VAL_EPS
            if above and s["last_above"]:
                s["alerts"].append(
                    f"anneal val worsening 2nd point step={step} val={val} "
                    f"(anneal min {s['ann_min']})")
            if not above:
                s["ann_min"] = min(s["ann_min"], val)
            s["last_above"] = above


def alert(msg):
    line = f"ALERT {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}"
    with open(OUT, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    print(line, flush=True)


def main():
    with open(OUT, "a", encoding="utf-8") as fh:
        fh.write(f"watch start {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
                 f"anneal_start={ANNEAL}\n")
    s = new_state()
    # rebuild history so a restart mid-anneal keeps running min, gnorm median, skip window
    try:
        with open(LOG, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                handle_line(s, line)
        pos = os.path.getsize(LOG)
        s["alerts"] = []  # history never re-alerts
    except OSError:
        pos = 0
    while True:
        try:
            size = os.path.getsize(LOG)
        except OSError:
            time.sleep(60)
            continue
        if size < pos:
            pos = 0
        if size > pos:
            with open(LOG, encoding="utf-8", errors="replace") as fh:
                fh.seek(pos)
                chunk = fh.read()
                pos = fh.tell()
            for line in chunk.splitlines():
                before = len(s["alerts"])
                handle_line(s, line)
                for a in s["alerts"][before:]:
                    alert(a)
        time.sleep(60)


def _selftest():
    def run(lines):
        s = new_state()
        for ln in lines:
            handle_line(s, ln)
        return s["alerts"]

    def L(step, phase, loss, g=0.2):
        return (f"step {step}/38070 90% [{phase}] | loss {loss} | lr 1e-03 | "
                f"gnorm {g} | 27K tok/s/gpu")

    # 1 cross-phase +1.0 at the main->anneal boundary: no jump alert
    a = run([L(34260, "main", 0.377), L(34270, "anneal", 1.368)])
    assert not [x for x in a if "jump" in x], a
    # 2 same-phase adjacent rise > 0.5 inside the window: alerts
    a = run([L(34300, "anneal", 0.40), L(34310, "anneal", 0.95)])
    assert any("jump" in x and "phase=anneal" in x for x in a), a
    # 3 REAL trainer literal: single skip is WARN, not CRASH
    a = run(["step 34300/38070 non-finite grad — step skipped (1)"])
    assert any("WARN" in x for x in a) and not [x for x in a if x.startswith("CRASH")], a
    # 4 REAL trainer literal: 10 skips in 500 steps escalates to CRASH
    a = run([f"step {34270 + i * 40}/38070 non-finite grad — step skipped ({i+1})"
             for i in range(10)])
    assert any(x.startswith("CRASH") and "non-finite skips" in x for x in a), a
    # 5 11 spread-out skips do NOT escalate (window prunes the oldest)
    a = run([f"step {34000 + i * 60}/38070 non-finite grad — step skipped ({i+1})"
             for i in range(11)])
    assert not [x for x in a if x.startswith("CRASH")], a
    # 6 REAL trainer literal: rollback is CRASH immediately
    a = run(["step 34300/38070 20 skips in a row — rolled back to snapshot"])
    assert any(x.startswith("CRASH") and "rollback" in x for x in a), a
    # 7 lowercase nan fallback
    a = run([L(34300, "anneal", "nan")])
    assert any(x.startswith("CRASH") and "NaN" in x for x in a), a
    # 8 gnorm spike after 5 calm points
    a = run([L(34270 + i, "anneal", 0.5, 0.20) for i in range(5)]
            + [L(34280, "anneal", 0.5, 2.00)])
    assert any("gnorm blow-up" in x for x in a), a
    # 9 single gnorm wobble at 4x stays silent
    a = run([L(34270 + i, "anneal", 0.5, 0.20) for i in range(5)]
            + [L(34280, "anneal", 0.5, 0.80)])
    assert not [x for x in a if "gnorm" in x], a
    # 10 worsening needs TWO points beyond min+0.005; a 0.003 wobble stays silent
    s = new_state()
    for step, val in [(34500, 1.700), (35000, 1.702)]:
        handle_line(s, f"step {step}/38070 val {val:.3f} val_s 4.0")
    assert not s["alerts"], s["alerts"]
    for step, val in [(35500, 1.710), (36000, 1.712)]:
        handle_line(s, f"step {step}/38070 val {val:.3f} val_s 4.0")
    assert [x for x in s["alerts"] if "worsening" in x], s["alerts"]
    # 11 single bounce below eps does not alert; a new min resets state
    s = new_state()
    for step, val in [(34500, 1.7), (35000, 1.6), (35500, 1.602)]:
        handle_line(s, f"step {step}/38070 val {val} val_s 4.0")
    assert not s["alerts"], s["alerts"]
    print("selftest ok: 11 cases")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        main()
