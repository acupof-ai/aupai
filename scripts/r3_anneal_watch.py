#!/usr/bin/env python3
"""r3 anneal watcher: crash-class signals only; trend alerts stay separate.

Alerts appended to runs/r3_anneal_watch.log:
- NaN: unconditional, any training line.
- anneal-head loss jump: adjacent SAME-PHASE ([main] vs [anneal]) lines within the
  head window whose loss differs by > 0.5. Cross-phase pairs are different domain
  batches and swing 1.0 routinely (2026-09-14 false alerts at step 34300).
- gnorm blow-up: latest head-window gnorm > 5x the rolling window median.
- anneal val worsening: two consecutive anneal val points above the running min.

Never touches the training process. `--selertest` runs synthetic lines through the
parser and asserts each alert class; no pod or log access.
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

LINE_RE = re.compile(r"step (\d+)/38070.*?(?:\d+% )?\[(main|anneal)\].*?\| loss ([0-9.]+|nan).*?gnorm ([0-9.]+)")
VAL_RE = re.compile(r"step (\d+)/38070 val ([0-9.]+)")


def new_state():
    return {
        "prev_step": 0, "prev_phase": None, "prev_loss": None,
        "gnorms": [], "val_seen": set(), "ann_min": None, "last_above": False,
        "alerts": [],
    }


def handle_line(s, line):
    if "NaN" in line and "step" in line:
        s["alerts"].append(f"NaN in log: {line.strip()[:200]}")
    m = LINE_RE.search(line)
    if m:
        step, phase, loss_s, g = int(m.group(1)), m.group(2), m.group(3), float(m.group(4))
        if ANNEAL <= step <= HEAD_END:
            s["gnorms"].append(g)
            if (s["prev_phase"] == phase and s["prev_loss"] is not None
                    and step != s["prev_step"] and loss_s != "nan"
                    and abs(float(loss_s) - s["prev_loss"]) > JUMP):
                s["alerts"].append(
                    f"anneal-head loss jump step={step} phase={phase} "
                    f"{s['prev_loss']}->{loss_s} d={float(loss_s) - s['prev_loss']:+.3f}")
            n = len(s["gnorms"])
            if n >= GNORM_MIN_N and g > GNORM_X * statistics.median(s["gnorms"][:-1]):
                s["alerts"].append(
                    f"gnorm blow-up step={step} gnorm={g:.3f} "
                    f"median={statistics.median(s['gnorms'][:-1]):.3f}")
            s["prev_step"], s["prev_phase"], s["prev_loss"] = step, phase, float(loss_s)
    v = VAL_RE.search(line)
    if v:
        step, val = int(v.group(1)), float(v.group(2))
        if step >= ANNEAL and step not in s["val_seen"]:
            s["val_seen"].add(step)
            if s["ann_min"] is None:
                s["ann_min"] = val
            above = val > s["ann_min"]
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
    # rebuild history so a restart mid-anneal keeps the running min and gnorm median
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
            with open(LOG, "r", encoding="utf-8", errors="replace") as fh:
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

    # 1 cross-phase +1.0 at the boundary: no jump alert
    a = run([L(34260, "main", 0.377), L(34270, "anneal", 1.368)])
    assert not [x for x in a if "jump" in x], a
    # 2 same-phase adjacent rise > 0.5 inside the window: alerts
    a = run([L(34300, "anneal", 0.40), L(34310, "anneal", 0.95)])
    assert any("jump" in x and "phase=anneal" in x for x in a), a
    # 3 NaN unconditional
    a = run([L(30000, "main", 0.5).replace("loss 0.5", "loss NaN")])
    assert any("NaN" in x for x in a), a
    # 4 gnorm spike after 5 calm points
    a = run([L(34270 + i, "anneal", 0.5, 0.20) for i in range(5)]
            + [L(34280, "anneal", 0.5, 2.00)])
    assert any("gnorm blow-up" in x for x in a), a
    # 5 single gnorm wobble under 5x stays silent
    a = run([L(34270 + i, "anneal", 0.5, 0.20) for i in range(5)]
            + [L(34280, "anneal", 0.5, 0.80)])
    assert not [x for x in a if "gnorm" in x], a
    # 6 worsening: two points above running min
    s = new_state()
    for step, val in [(34500, 1.7), (35000, 1.6), (35500, 1.65), (36000, 1.7)]:
        handle_line(s, f"step {step}/38070 val {val} val_s 4.0")
    assert [x for x in s["alerts"] if "worsening" in x], s["alerts"]
    # 7 single bounce does not alert, new min resets
    s = new_state()
    for step, val in [(34500, 1.7), (35000, 1.65)]:
        handle_line(s, f"step {step}/38070 val {val} val_s 4.0")
    assert not s["alerts"], s["alerts"]
    print("selftest ok: 7 cases")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        main()
