#!/usr/bin/env python3
"""Write and read runs/memory_diag.jsonl -- readout 4 of memory_layers_0905.

train.py calls `log_diag(...)` every 100 steps on rank 0. The schema is
data/ledger_schema.json and this module VALIDATES against it rather than restating
it, so a field that drifts is a refusal at the write and not a column of nulls
discovered when someone plots the curves.

Why validate at all, when the caller is one line in train.py: the two stop rules in
the charter read this file (pool_touched < 0.20 at step 1000; tok/s/gpu < 70K at step
30). A row missing tok_s_gpu makes the monitor silent, and silence is what the stop
rule looks like when nothing is wrong. The failure this prevents is not a bad number,
it is a rule that cannot fire.

`name` is the ARM (m1, m2, m3), not the run kind. Readout 3 compares M2 against M1
against the control and 98 draws three curves, so a row without the arm folds three
lines into one, unrecoverably -- the file is append-only and nothing else records
which arm a row came from.

Usage:
  python scripts/memory_diag.py --selftest

# restartable: every write is one O_APPEND line and the ledger is append-only, so an
# interrupt costs at most the row in flight -- the next 100-step tick writes the next row
# and nothing needs to be replayed. There is no partial state to resume: readouts 4 and 5
# read the newest row per arm, not a cumulative total.
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCHEMA_PATH = os.path.join(ROOT, "data", "ledger_schema.json")
LEDGER = "runs/memory_diag.jsonl"

# Charter thresholds (docs/standards/memory_layers_0905.md readouts 4 and 5). Here so the
# monitor and the harness check read one definition; the charter is the authority and this
# is a transcription -- if they disagree the charter wins and this is the bug.
STOP_TOK_S_GPU = 70_000.0
STOP_TOK_S_AT_STEP = 30
STOP_POOL_TOUCHED = 0.20
STOP_POOL_AT_STEP = 1000
# A running arm must have a row this recent. 100-step cadence, so 300 allows two missed
# writes before the check goes red -- one missed write is a slow save, three is a dead hook.
FRESH_WITHIN_STEPS = 300
# The commit that made train.py write rows at steps 10/20/30 as well as every 100. A run
# whose own commit does NOT have this as an ancestor could not have written an early row, so
# an empty ledger for it is expected rather than a defect; from this commit on, an arm that
# passed step 10 and left no row had a dead hook. Recorded as a sha rather than a date because
# a branch can carry an older tree at a later wall-clock time.
DIAG_CADENCE_COMMIT = "6b678541"


class DiagRowInvalid(ValueError):
    """A row does not match the schema, so it is refused rather than written.

    Refused, not coerced: readout 4's stop rule cannot fire on a field that is absent,
    and a partially-written ledger reads exactly like a healthy one that has not
    collapsed yet."""


def schema(path=None):
    with open(path or SCHEMA_PATH, encoding="utf-8") as f:
        return json.load(f)["memory_diag"]


def validate(row, sch=None):
    """Return the row, or raise DiagRowInvalid naming every problem at once.

    Every problem, not the first: a caller fixing one field at a time reruns a
    100-step training loop per mistake."""
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
        # A bool is not a number here even though Python says it is: `True` for
        # tok_s_gpu would validate as 1.0 and read as a stopped arm.
        if isinstance(v, bool):
            bad.append(f"{k}: want {want}, got bool ({v!r})")
            continue
        if want == "int" and not isinstance(v, int):
            bad.append(f"{k}: want int, got {type(v).__name__} ({v!r})")
            continue
        # An int IS an acceptable float: a caller computing a ratio that lands exactly on
        # 0 or 1 passes an int, and refusing that would refuse the two most interesting
        # values -- a fully collapsed and a fully uniform pool.
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
        # (tok_s_per_gpu for tok_s_gpu), and ignoring it writes a row that passes
        # validation with the field the stop rule needs still absent.
        bad.append(f"unknown field(s): {', '.join(sorted(unknown))}")
    if bad:
        raise DiagRowInvalid("; ".join(bad))
    return row


def log_diag(name, step, pool_touched_frac, topk_entropy, key_gini, tok_s_gpu,
             n_values=None, topk=None, rows_changed_since_prev=None, rows_changed=None,
             root=None, path=None):
    """Append one row. Called from train.py at diag steps, rank 0 only."""
    row = {"name": name, "step": int(step),
           "pool_touched_frac": float(pool_touched_frac),
           "topk_entropy": float(topk_entropy),
           "key_gini": float(key_gini),
           "tok_s_gpu": float(tok_s_gpu),
           "ts": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())}
    if n_values is not None:
        row["n_values"] = int(n_values)
    if topk is not None:
        row["topk"] = int(topk)
    # READOUT 6, both optional and OMITTED rather than defaulted when absent. The first diag step
    # of a run has no previous checksum to compare against, and writing 0.0 there would be the
    # same number a completely frozen table produces -- the one reading this field exists to
    # distinguish. An absent field reads as "not measured yet"; a 0.0 reads as "measured, nothing
    # moved", and only one of those is true at step 10.
    if rows_changed_since_prev is not None:
        row["rows_changed_since_prev"] = float(rows_changed_since_prev)
    if rows_changed is not None:
        row["rows_changed"] = int(rows_changed)
    validate(row)
    p = path or os.path.join(root or ROOT, LEDGER)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    # One write() of one complete line under O_APPEND, the same shape as harness's
    # _append_task: concurrent appends below a page do not interleave and no reader sees a
    # partial row. Built first because f.write() of a str can flush at a buffer boundary.
    line = (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, line)
    finally:
        os.close(fd)
    return row


def read_rows(root=None, path=None):
    """Every row, in file order. A malformed line is REPORTED, never skipped.

    Returns (rows, errors). A skip would make a corrupt ledger read as a short one, and
    the check below turns "no rows" into "the arm is not logging" -- the opposite
    conclusion from "the ledger is damaged"."""
    p = path or os.path.join(root or ROOT, LEDGER)
    rows, errs = [], []
    if not os.path.exists(p):
        return rows, errs
    with open(p, encoding="utf-8") as f:
        for i, ln in enumerate(f, 1):
            if not ln.strip():
                continue
            try:
                rows.append(json.loads(ln))
            except ValueError as e:
                errs.append(f"line {i}: {e}")
    return rows, errs


def latest_by_arm(rows):
    """{arm name: the row with the highest step}. Ties take the last written."""
    out = {}
    for r in rows:
        nm = r.get("name")
        if not isinstance(nm, str) or not nm:
            continue
        st = r.get("step")
        if not isinstance(st, int):
            continue
        cur = out.get(nm)
        if cur is None or st >= cur.get("step", -1):
            out[nm] = r
    return out


def stop_rules(row):
    """Which pre-registered stop rules this row trips, as (rule, evidence) pairs.

    The monitor WARNs and messages; it does not kill. Empty list is the normal case."""
    hit = []
    st = row.get("step")
    if not isinstance(st, int):
        return hit
    tok = row.get("tok_s_gpu")
    # ">= the step", not "== the step": rows land on the 100-step cadence, so a rule
    # written at step 30 never sees a row numbered 30 and an equality test never fires.
    if isinstance(tok, (int, float)) and st >= STOP_TOK_S_AT_STEP and tok < STOP_TOK_S_GPU:
        hit.append(("readout 5: tok/s/gpu below 70K",
                    f"{tok:.0f} tok/s/gpu at step {st} (control 82K, stop below "
                    f"{STOP_TOK_S_GPU:.0f})"))
    pt = row.get("pool_touched_frac")
    if isinstance(pt, (int, float)) and st >= STOP_POOL_AT_STEP and pt < STOP_POOL_TOUCHED:
        hit.append(("readout 4: pool collapse",
                    f"pool_touched_frac {pt:.3f} at step {st} (stop below "
                    f"{STOP_POOL_TOUCHED})"))
    return hit


def check_arm(name, root=None, path=None):
    """One line saying whether this arm's newest row trips a stop rule, and exit 0/1/2.

    The launch monitor's reader. It exists so the monitor does not re-transcribe the two
    thresholds: a monitor carrying its own copy of 70000 and 0.20 is a second definition
    that drifts from the charter silently, and the drift is invisible because both numbers
    look right in isolation.

    Exit 0 nothing tripped, 1 a rule tripped, 2 nothing to read yet. 2 is not an error:
    before step 100 an arm legitimately has no row, and a monitor must not treat that as a
    stop.
    """
    rows, errs = read_rows(root=root, path=path)
    if errs:
        print(f"memory_diag: {len(errs)} unreadable line(s): {errs[0]}")
        return 1
    want = _arm_key(name)
    latest = latest_by_arm(rows)
    key = next((k for k in latest if _arm_key(k) == want), None)
    if key is None:
        print(f"memory_diag: no row for arm {want or name} yet")
        return 2
    row = latest[key]
    hits = stop_rules(row)
    if not hits:
        print(f"memory_diag: {want} step {row.get('step')} within bounds "
              f"(pool {row.get('pool_touched_frac')}, {row.get('tok_s_gpu')} tok/s/gpu)")
        return 0
    for rule, why in hits:
        print(f"STOP RULE {rule} -- {why}")
    return 1


def _arm_key(name):
    """'m1' from b0_mem_m1, m1_probe or m1; None from a non-arm name.

    The join key between a run name and its rows. Both sides are written by different
    hands -- the launch names the run, train.py's hook names the row -- so neither is a
    prefix of the other: b0_mem_m1 against m1 is the live case, and a prefix test returns
    False for it."""
    import re

    m = re.search(r"(^|_)mem_m([123])([_-]|$)|(^|_)m([123])([_-]|$)", str(name or ""), re.I)
    if not m:
        return None
    return "m" + (m.group(2) or m.group(5))


def _selftest():
    import shutil
    import tempfile

    sch = schema()
    assert sch["path"] == LEDGER, sch["path"]
    for req in ("name", "step", "pool_touched_frac", "topk_entropy", "key_gini", "tok_s_gpu"):
        assert sch["fields"][req]["required"], f"{req} must be required"

    d = tempfile.mkdtemp(prefix="memdiag_st_")
    try:
        p = os.path.join(d, "runs", "memory_diag.jsonl")

        # 1. A good row round-trips, and the arm name survives -- the field the whole
        #    ledger exists for.
        log_diag("m1", 100, 0.86, 3.2, 0.31, 81500.0, n_values=1048576, topk=32, path=p)
        rows, errs = read_rows(path=p)
        assert not errs and len(rows) == 1, (rows, errs)
        assert rows[0]["name"] == "m1" and rows[0]["step"] == 100, rows[0]
        assert rows[0]["n_values"] == 1048576 and "ts" in rows[0], rows[0]

        # 2. Three arms do not fold together. This is the reason `name` is required: a
        #    ledger keyed on step alone gives one curve where the charter needs three.
        log_diag("m2", 100, 0.91, 3.1, 0.28, 82200.0, path=p)
        log_diag("m1", 200, 0.84, 3.2, 0.33, 81000.0, path=p)
        rows, _ = read_rows(path=p)
        latest = latest_by_arm(rows)
        assert set(latest) == {"m1", "m2"}, latest
        assert latest["m1"]["step"] == 200 and latest["m2"]["step"] == 100, latest

        # 3. Every invalid row is REFUSED and nothing is appended. Known answers, one per
        #    failure mode, and the count must not move.
        n_before = len(read_rows(path=p)[0])
        cases = [
            ({"name": "", "step": 1, "pool_touched_frac": 0.5, "topk_entropy": 1.0,
              "key_gini": 0.1, "tok_s_gpu": 80000.0}, "empty arm name"),
            ({"step": 1, "pool_touched_frac": 0.5, "topk_entropy": 1.0,
              "key_gini": 0.1, "tok_s_gpu": 80000.0}, "no arm name at all"),
            ({"name": "m1", "step": 1, "pool_touched_frac": 1.5, "topk_entropy": 1.0,
              "key_gini": 0.1, "tok_s_gpu": 80000.0}, "fraction above 1"),
            ({"name": "m1", "step": 1, "pool_touched_frac": -0.1, "topk_entropy": 1.0,
              "key_gini": 0.1, "tok_s_gpu": 80000.0}, "fraction below 0"),
            ({"name": "m1", "step": 1, "pool_touched_frac": 0.5, "topk_entropy": 1.0,
              "key_gini": 0.1}, "tok_s_gpu absent -- readout 5 could not fire"),
            # EVERY required field present, so the ONLY defect is the extra key. The
            # earlier version of this case omitted tok_s_gpu as well, so it passed on the
            # required-field rule and said nothing about unknown keys -- an assertion
            # testing something adjacent to what it claimed. Caught by mutation 2026-09-05.
            ({"name": "m1", "step": 1, "pool_touched_frac": 0.5, "topk_entropy": 1.0,
              "key_gini": 0.1, "tok_s_gpu": 80000.0, "tok_s_per_gpu": 80000.0},
             "typo BESIDE a complete row: unknown key must refuse on its own"),
            # True is an int in Python, so an unguarded isinstance check reads it as 1.0 --
            # a tok_s_gpu of True would validate and look like a stopped arm.
            ({"name": "m1", "step": 1, "pool_touched_frac": 0.5, "topk_entropy": 1.0,
              "key_gini": 0.1, "tok_s_gpu": True}, "bool as tok_s_gpu"),
            ({"name": "m1", "step": True, "pool_touched_frac": 0.5, "topk_entropy": 1.0,
              "key_gini": 0.1, "tok_s_gpu": 80000.0}, "bool as step"),
            ({"name": "m1", "step": "100", "pool_touched_frac": 0.5, "topk_entropy": 1.0,
              "key_gini": 0.1, "tok_s_gpu": 80000.0}, "step as a string"),
        ]
        for bad_row, label in cases:
            try:
                validate(bad_row)
            except DiagRowInvalid:
                pass
            else:
                raise AssertionError(f"validate accepted a bad row: {label} {bad_row}")
        assert len(read_rows(path=p)[0]) == n_before, "a refused row was written anyway"

        # 4. A row that is fine must PASS. Without this the suite passes on a validate()
        #    that refuses everything -- the shape that shipped three times in this repo.
        validate({"name": "m3", "step": 0, "pool_touched_frac": 0.0, "topk_entropy": 0.0,
                  "key_gini": 0.0, "tok_s_gpu": 0.0})
        validate({"name": "m3", "step": 1, "pool_touched_frac": 1, "topk_entropy": 3.4657,
                  "key_gini": 1, "tok_s_gpu": 82000})

        # 5. The stop rules fire on the charter's own numbers, and only then.
        assert stop_rules({"name": "m1", "step": 100, "tok_s_gpu": 69000.0,
                           "pool_touched_frac": 0.9}), "readout 5 did not fire at 69K"
        assert not stop_rules({"name": "m1", "step": 100, "tok_s_gpu": 81000.0,
                               "pool_touched_frac": 0.9}), "readout 5 fired on a healthy row"
        # Below threshold but BEFORE the step the rule names: warmup, not a stop.
        assert not stop_rules({"name": "m1", "step": 10, "tok_s_gpu": 40000.0,
                               "pool_touched_frac": 0.9}), "readout 5 fired during warmup"
        assert not stop_rules({"name": "m1", "step": 500, "tok_s_gpu": 81000.0,
                               "pool_touched_frac": 0.05}), "readout 4 fired before step 1000"
        hits = stop_rules({"name": "m1", "step": 1000, "tok_s_gpu": 81000.0,
                           "pool_touched_frac": 0.05})
        assert len(hits) == 1 and "collapse" in hits[0][0], hits
        both = stop_rules({"name": "m1", "step": 1000, "tok_s_gpu": 50000.0,
                           "pool_touched_frac": 0.05})
        assert len(both) == 2, both

        # 6. A corrupt line is reported, not skipped. A skip makes a damaged ledger read
        #    as a short one, and "no rows" is the opposite diagnosis from "damaged".
        with open(p, "a", encoding="utf-8") as f:
            f.write("{not json\n")
        rows, errs = read_rows(path=p)
        assert len(errs) == 1 and "line" in errs[0], errs
        assert len(rows) == n_before, "a corrupt line changed the row count"

        # 7. No ledger yet is (no rows, no errors) -- distinct from a corrupt one.
        rows, errs = read_rows(path=os.path.join(d, "nope.jsonl"))
        assert rows == [] and errs == [], (rows, errs)
        # 8. check_arm is the monitor's reader, and its three exit codes are the contract:
        #    0 within bounds, 1 a rule tripped, 2 nothing to read yet. 2 must not be an
        #    error -- an arm before step 100 has no row, and a monitor treating that as a
        #    stop would kill every launch in its first minutes.
        p2 = os.path.join(d, "runs", "arm.jsonl")
        assert check_arm("b0_mem_m1", path=p2) == 2, "no ledger must exit 2, not 1"
        log_diag("m1", 500, 0.83, 3.1, 0.30, 81000.0, path=p2)
        # THE LIVE SPELLING: run named b0_mem_m1, rows named m1. Neither is a prefix of the
        # other, which is the join that broke in harness's copy of this logic.
        assert check_arm("b0_mem_m1", path=p2) == 0, "healthy arm must exit 0"
        assert check_arm("m1", path=p2) == 0, "bare arm id must resolve too"
        assert check_arm("b0_mem_m2", path=p2) == 2, "a different arm has no row: exit 2"
        log_diag("m1", 600, 0.83, 3.1, 0.30, 41000.0, path=p2)
        assert check_arm("b0_mem_m1", path=p2) == 1, "tripped readout 5 must exit 1"
        for nm, want in (("b0_mem_m1", "m1"), ("m3_probe", "m3"), ("mem_m2_x", "m2"),
                         ("p200m_control", None), ("b0_memoir_m1x", None),
                         ("b0_mem_m4", None)):
            assert _arm_key(nm) == want, f"_arm_key({nm}) = {_arm_key(nm)}, want {want}"

    finally:
        shutil.rmtree(d, ignore_errors=True)

    print("memory_diag selftest OK (8 cases: round-trip, arms separate, 9 refusals, "
          "positive control, stop rules incl. warmup, corrupt line, absent file, "
          "check_arm exit codes on the live spelling)")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    if "--check-arm" in sys.argv:
        sys.exit(check_arm(sys.argv[sys.argv.index("--check-arm") + 1]))
    sys.exit(0)
