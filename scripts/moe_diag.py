#!/usr/bin/env python3
"""Write and read runs/moe_diag.jsonl -- readout 4 of moe_0905.

train.py calls `log_diag(...)` every 100 steps on rank 0. The schema is
data/ledger_schema.json under "moe_diag" and this module VALIDATES against it rather than
restating it, so a field that drifts is a refusal at the write rather than a column of nulls
found when someone plots the curves.

WHY VALIDATE, when the caller is one line in train.py: readout 4's two stop rules read this file
(usage_frac < 0.50 at step 300; entropy_norm < 0.70 at step 500). A row missing a field makes the
monitor silent, and silence is what a stop rule looks like when nothing is wrong. The failure this
prevents is not a bad number -- it is a rule that cannot fire.

`name` IS THE ARM (e1, e1b), not the run kind. E1b is the layer-count axis against E1, so a row
without the arm folds two curves into one, unrecoverably: the file is append-only and nothing else
records which arm a row came from.

ROW IDENTITY IS (name, step, ts), NOT (name, step). Ruled from the memory ledger's own measurement
-- an arm relaunched under the same name restarts at step 10 and re-visits every step the previous
run wrote, and on b0_mem_m1's relaunch (name, step) collided three ways while (name, step, ts)
gave 28 distinct over 28 rows. A MoE arm relaunched after a stop rule fires is the EXPECTED case,
so the cheaper key is wrong from day one.

THIS FILE LANDS WITH ITS THREE REGISTRATIONS, in one commit, deliberately. runs/memory_diag.jsonl
existed for a full DAY with a writer and no transport in either direction: absent from
ledger_audit.KEYS (which is what pod_pull_ledgers derives its file list from) and excluded by
pod_push.sh, so 25 rows behind three stopped arms lived only on the pod and rescuing them needed a
deliberate-exception commit. The three are: the schema entry, the .gitattributes union-merge line,
and the ledger_audit KEYS entry.

Usage:
  python scripts/moe_diag.py --selftest

# restartable: every write is one O_APPEND line and the ledger is append-only, so an interrupt
# costs at most the row in flight -- the next 100-step tick writes the next row and nothing needs
# replaying. Readout 4 reads the newest row per arm, not a cumulative total.
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCHEMA_PATH = os.path.join(ROOT, "data", "ledger_schema.json")
LEDGER = "runs/moe_diag.jsonl"

# Charter thresholds (docs/standards/moe_0905.md readout 4, and amendment 1 of the prereg row).
# Here so the monitor and the harness check read ONE definition; the charter is the authority and
# this is a transcription -- if they disagree the charter wins and this is the bug.
#
# THE ENTROPY STEP IS 500, NOT 1000, and that number is the reason this comment exists: my first
# prereg row said 1000 while the charter said 500, 44 caught it, and amendment 1 corrected the row.
# The charter's argument is the cost: M1's collapse was monotone by step 300 (pool_touched_frac
# 0.137) and 700 further steps on two cards only confirmed what the first three windows showed.
STOP_USAGE_FRAC = 0.50
STOP_USAGE_AT_STEP = 300
STOP_ENTROPY_NORM = 0.70
STOP_ENTROPY_AT_STEP = 500
# A running arm must have a row this recent. 100-step cadence, so 300 allows two missed writes
# before the check goes red -- one missed write is a slow save, three is a dead hook.
FRESH_WITHIN_STEPS = 300


class DiagRowInvalid(ValueError):
    """A row does not match the schema, so it is refused rather than written.

    Refused, not coerced: readout 4's stop rules cannot fire on a field that is absent, and a
    partially-written ledger reads exactly like a healthy one that has not collapsed yet."""


def schema(path=None):
    with open(path or SCHEMA_PATH, encoding="utf-8") as f:
        return json.load(f)["moe_diag"]


def validate(row, sch=None):
    """Return the row, or raise DiagRowInvalid naming every problem at once.

    Every problem, not the first: a caller fixing one field at a time reruns a 100-step training
    loop per mistake."""
    sch = sch or schema()
    fields = sch["fields"]
    bad = []
    for k, spec in fields.items():
        if k not in row:
            if spec.get("required"):
                bad.append(f"{k}: required and absent")
            continue
        v = row[k]
        want = spec["type"]
        # A bool is not a number here even though Python says it is: `True` for usage_frac would
        # validate as 1.0 and read as a perfectly spread router.
        if isinstance(v, bool):
            bad.append(f"{k}: want {want}, got bool ({v!r})")
            continue
        if want == "int" and not isinstance(v, int):
            bad.append(f"{k}: want int, got {type(v).__name__} ({v!r})")
            continue
        # An int IS an acceptable float: a fraction landing exactly on 0 or 1 arrives as an int,
        # and refusing that would refuse the two most interesting values -- a fully collapsed and
        # a perfectly uniform router.
        if want == "float" and not isinstance(v, (int, float)):
            bad.append(f"{k}: want float, got {type(v).__name__} ({v!r})")
            continue
        if want == "str" and not isinstance(v, str):
            bad.append(f"{k}: want str, got {type(v).__name__} ({v!r})")
            continue
        if want == "str" and not v.strip():
            bad.append(f"{k}: empty")
        if "min" in spec and isinstance(v, (int, float)) and v < spec["min"]:
            bad.append(f"{k}: {v} below min {spec['min']}")
        if "max" in spec and isinstance(v, (int, float)) and v > spec["max"]:
            bad.append(f"{k}: {v} above max {spec['max']}")
    unknown = set(row) - set(fields)
    if unknown:
        # Refused, not ignored: an unknown key is nearly always a typo in a required one
        # (usage_fraction for usage_frac), and ignoring it writes a row that passes validation
        # with the field the stop rule needs still absent.
        bad.append(f"unknown field(s): {', '.join(sorted(unknown))}")
    if bad:
        raise DiagRowInvalid("; ".join(bad))
    return row


def log_diag(name, step, usage_frac, entropy_norm, load_gini, tokens, window_steps,
             n_routed=None, top_k=None, tok_s_gpu=None, root=None, path=None):
    """Append one row. Called from train.py at diag steps, rank 0 only.

    window_steps is POSITIONAL AND REQUIRED, not a keyword with a default. A default would let a
    caller omit it and still write a row that validates, which is exactly the omission 44's review
    made a blocking condition -- a fraction whose window length is unknown is not comparable to
    another row's.
    """
    row = {"name": name, "step": int(step),
           "usage_frac": float(usage_frac),
           "entropy_norm": float(entropy_norm),
           "load_gini": float(load_gini),
           "tokens": int(tokens),
           "window_steps": int(window_steps),
           "ts": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())}
    for k, v, cast in (("n_routed", n_routed, int), ("top_k", top_k, int),
                       ("tok_s_gpu", tok_s_gpu, float)):
        # OMITTED rather than defaulted when absent, the memory ledger's rule for readout 6: a
        # 0.0 written for an unknown throughput reads exactly like a stopped arm, and readout 5 is
        # only defined at steps 30 and 100 anyway.
        if v is not None:
            row[k] = cast(v)
    validate(row)
    p = path or os.path.join(root or ROOT, LEDGER)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    # One write() of one complete line under O_APPEND, the same shape as harness's _append_task:
    # concurrent appends below a page do not interleave and no reader sees a partial row. Built
    # first because f.write() of a str can flush at a buffer boundary.
    line = (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, line)
    finally:
        os.close(fd)
    return row


def read_rows(root=None, path=None):
    """Every row, oldest first. Malformed lines are REPORTED, not skipped."""
    p = path or os.path.join(root or ROOT, LEDGER)
    if not os.path.exists(p):
        return []
    out = []
    with open(p, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except ValueError as e:
                raise DiagRowInvalid(f"{LEDGER}:{i} is not JSON: {e}") from e
    return out


def stop_rules(rows):
    """Which pre-registered stop rules have fired, re-derived from the ledger.

    RE-DERIVED FROM THE REPO rather than transcribed into a report: the thresholds live at the top
    of this file, the charter is their authority, and a reader can rerun this instead of trusting
    a number someone typed. Returns one dict per arm.

    THE COMPARISON IS >= THE STEP, not == it. A run whose diag cadence skipped the exact step (a
    resume landing at 305, a save stalling the 300 tick) would otherwise never be judged, and
    "the rule never fired" is indistinguishable from "the rule could not fire".
    """
    by_arm = {}
    for r in rows:
        by_arm.setdefault(r.get("name", "?"), []).append(r)
    out = {}
    for arm, rs in by_arm.items():
        rs = sorted(rs, key=lambda r: (int(r.get("step", 0)), str(r.get("ts", ""))))
        fired, checked = [], []
        for label, field, thresh, at_step, cmp_lo in (
            ("usage_frac", "usage_frac", STOP_USAGE_FRAC, STOP_USAGE_AT_STEP, True),
            ("entropy_norm", "entropy_norm", STOP_ENTROPY_NORM, STOP_ENTROPY_AT_STEP, True),
        ):
            cand = [r for r in rs if int(r.get("step", 0)) >= at_step and field in r]
            if not cand:
                continue
            r = cand[0]
            checked.append(f"{label} at step {r['step']} (rule step {at_step})")
            v = float(r[field])
            if (v < thresh) if cmp_lo else (v > thresh):
                fired.append(f"{label} {v:.4f} below {thresh} at step {r['step']}")
        out[arm] = {"fired": fired, "checked": checked,
                    "rows": len(rs),
                    "last_step": int(rs[-1]["step"]) if rs else None}
    return out


def _selftest():
    """Known-answer worlds. Each case states what it would catch."""
    import tempfile
    bad = 0

    def check(label, ok, detail=""):
        nonlocal bad
        bad += 0 if ok else 1
        print(f"  {'ok  ' if ok else 'BUG '} {label}{'' if ok else ' -- ' + detail}")

    good = dict(name="e1", step=300, usage_frac=0.62, entropy_norm=0.81, load_gini=0.22,
                tokens=7_864_320, window_steps=100)

    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "runs", "moe_diag.jsonl")
        row = log_diag(path=p, n_routed=24, top_k=3, **good)
        check("a complete row writes and validates", os.path.exists(p) and row["step"] == 300)
        # ts IS WRITTEN BY THE WRITER, not accepted from the caller: row identity depends on it,
        # and a caller-supplied ts could collide by being copied between two rows.
        check("the writer stamps ts itself", bool(row.get("ts")), f"row={row}")

        # window_steps CANNOT BE OMITTED. It is positional, so this is a TypeError rather than a
        # validation failure -- which is the stronger guarantee: the row cannot be constructed.
        try:
            log_diag("e1", 300, 0.62, 0.81, 0.22, 7_864_320, path=p)  # noqa: PLE1120
            check("window_steps cannot be omitted", False, "the call succeeded")
        except TypeError:
            check("window_steps cannot be omitted (positional, so it is a TypeError)", True)

        # A MISSING REQUIRED FIELD IS REFUSED. Written as a direct validate() call because the
        # writer's signature already forces the required ones -- this covers a future caller that
        # builds a row itself.
        for miss in ("usage_frac", "entropy_norm", "load_gini", "tokens", "window_steps", "name"):
            r = {k: v for k, v in good.items() if k != miss}
            r["ts"] = "2026-09-05 00:00:00"
            try:
                validate(r)
                check(f"a row missing {miss} is refused", False, "validate accepted it")
            except DiagRowInvalid:
                pass
        check("a row missing any required field is refused", True)

        # A BOOL IS NOT A FLOAT. True for usage_frac would validate as 1.0 and read as a
        # perfectly spread router -- the healthiest possible value, from a type error.
        r = dict(good, usage_frac=True, ts="2026-09-05 00:00:00")
        try:
            validate(r)
            check("usage_frac=True is refused", False, "validate accepted a bool")
        except DiagRowInvalid:
            check("usage_frac=True is refused (it would read as a perfect router)", True)

        # A TYPO IN A FIELD NAME IS REFUSED, not ignored. Ignoring it writes a row that passes
        # validation with the field the stop rule needs still absent.
        r = dict(good, ts="2026-09-05 00:00:00")
        r["usage_fraction"] = r.pop("usage_frac")
        try:
            validate(r)
            check("a typo'd field name is refused", False, "validate accepted it")
        except DiagRowInvalid:
            check("a typo'd field name is refused rather than ignored", True)

        # OUT-OF-RANGE IS REFUSED: a fraction above 1 is a counting bug upstream, and it would
        # make every stop rule read as healthy.
        try:
            validate(dict(good, usage_frac=1.4, ts="2026-09-05 00:00:00"))
            check("usage_frac 1.4 is refused", False, "validate accepted it")
        except DiagRowInvalid:
            check("usage_frac above 1.0 is refused", True)

    # THE STOP RULES, on known-answer worlds. The negative case matters as much as the positive:
    # a rule that fires on healthy numbers is as useless as one that never fires.
    healthy = [dict(name="e1", step=s, usage_frac=0.9, entropy_norm=0.95, load_gini=0.1,
                    tokens=1000, window_steps=100, ts=f"2026-09-05 00:0{i}:00")
               for i, s in enumerate((100, 200, 300, 400, 500))]
    sick = [dict(r, usage_frac=0.30, entropy_norm=0.40) for r in healthy]
    check("healthy rows fire nothing", not stop_rules(healthy)["e1"]["fired"],
          str(stop_rules(healthy)))
    fired = stop_rules(sick)["e1"]["fired"]
    check("collapsed rows fire BOTH rules", len(fired) == 2, str(fired))

    # THE >= COMPARISON. A run whose cadence skipped the exact rule step must still be judged, or
    # "never fired" and "could not fire" become the same reading.
    skewed = [dict(name="e1", step=s, usage_frac=0.30, entropy_norm=0.40, load_gini=0.1,
                   tokens=1000, window_steps=100, ts=f"2026-09-05 00:0{i}:00")
              for i, s in enumerate((305, 512))]
    check("a cadence that skipped the exact step is still judged",
          len(stop_rules(skewed)["e1"]["fired"]) == 2, str(stop_rules(skewed)))

    # TWO ARMS DO NOT FOLD INTO ONE CURVE. e1 sick, e1b healthy: the sick one must fire and the
    # healthy one must not, from the same ledger.
    mixed = sick + [dict(r, name="e1b") for r in healthy]
    sr = stop_rules(mixed)
    check("two arms are kept apart",
          bool(sr["e1"]["fired"]) and not sr["e1b"]["fired"], str(sr))

    # THE THRESHOLDS MATCH THE CHARTER, checked against the file rather than restated. This is the
    # transcription that was wrong once: my prereg row said the entropy rule fires at step 1000
    # while the charter said 500.
    ch = os.path.join(ROOT, "docs", "standards", "moe_0905.md")
    if os.path.exists(ch):
        txt = open(ch, encoding="utf-8").read()
        want = [f"below {STOP_USAGE_FRAC:.2f} at step {STOP_USAGE_AT_STEP}",
                f"below {STOP_ENTROPY_NORM:.2f} at\n   step {STOP_ENTROPY_AT_STEP}"]
        found = [w for w in want if w in txt]
        check(f"the charter states both thresholds this file transcribes ({len(found)}/2)",
              len(found) == 2,
              f"missing {[w for w in want if w not in found]} -- the charter is the authority, "
              f"so a mismatch means this file is the bug")
    else:
        check("the charter exists to check the thresholds against", False, f"{ch} absent")

    print(f"moe_diag selftest: {'PASS' if not bad else f'{bad} FAILED'}")
    return 1 if bad else 0


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        sys.exit(_selftest())
    sys.exit(f"usage: {os.path.basename(__file__)} --selftest  (got {sys.argv[1:]})")
