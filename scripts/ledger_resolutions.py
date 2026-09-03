#!/usr/bin/env python3
"""Adjudicated pod/local ledger disagreements (de-36 follow-up).

A key whose CURRENT row differs between the pod and the repository cannot be resolved by a
predicate: two closes of one event, written by different readers from different evidence.
Someone reads them and rules. This file records those rulings so
`harness check pod_ledger_rows_home` stops re-reporting a settled key -- a WARN that can
never be cleared is the same as no signal, and it teaches people to ignore the check.

    python3 scripts/ledger_resolutions.py            # the ledger, one line per ruling
    python3 scripts/ledger_resolutions.py --selftest

A RULING IS BOUND TO THE CONTENT IT WAS MADE ABOUT, not to the key. Every row carries
`local_fp` and `pod_fp`, sha256 over the sorted (field, value) pairs of the row each side
held when the ruling was issued. If either side later writes a new row under that key, the
fingerprints stop matching and the key WARNs again, saying the ruling was about a different
version. Without this a ruling is a permanent mute: the disagreement it settled is gone, and
the next one under the same key is silent. Three incidents bought this rule already
(vocab_id, .srcfp, filters_fp) -- a derived artifact must carry the fingerprint of what
produced it, and a ruling is derived from two rows.

`winner`:
  local   the local row stands; the pod row adds nothing. Typically the pod row is a
          monitor's mechanical close (process gone / log silent / interrupted, no result /
          an expired `running`) and the local row is a close written from the artifact.
  pod     the pod row stands.
  merged  the local row's STATUS stands, and the pod row's result/finding text is folded
          into the local row's finding. For a pod row that is the same verdict written more
          completely -- three p200m_4b_0902 rows carried val 2.569, MFU 32% and peak
          49.5 GiB that the local rows did not. `winner=local` there would have deleted
          measurements, which is why this class exists.
"""

import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(ROOT, "runs", "ledger_resolutions.jsonl")
WINNERS = ("local", "pod", "merged")


def fingerprint(row):
    """sha256 over the row's sorted (field, value) pairs, first 16 hex.

    Sorted so field order in the file cannot change it, and over the WHOLE row rather than
    the fields a reader compared: a ruling about a pair of rows is invalidated by any edit
    to either, including one to a field nobody looked at."""
    if row is None:
        return None
    payload = json.dumps(sorted((k, row[k]) for k in row), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def load(path=PATH):
    """[(ledger, key_tuple, row)] -- key as a tuple so it compares to a keyfn's output."""
    out = []
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            k = r.get("key")
            out.append((r.get("ledger"), tuple(k) if isinstance(k, list) else k, r))
    return out


def index(path=PATH):
    """{(ledger, key): row} for lookup by the check and the puller."""
    return {(led, key): r for led, key, r in load(path)}


def settled(ledger, key, local_row, pod_row, idx=None):
    """(is_settled, why). why is None when settled, and names the reason when not.

    A ruling settles the key ONLY while both sides still hold the rows it was made about.
    The stale case returns a reason rather than False alone, because "no ruling" and "the
    ruling was about other rows" are different situations for the reader: the first needs a
    decision, the second needs a re-decision and says what changed."""
    idx = index() if idx is None else idx
    r = idx.get((ledger, tuple(key) if isinstance(key, list) else key))
    if r is None:
        return False, None
    lf, pf = fingerprint(local_row), fingerprint(pod_row)
    if r.get("local_fp") != lf:
        return False, (f"ruled {r.get('date')} by {r.get('ruled_by')} about a different LOCAL "
                       f"row ({r.get('local_fp')}, now {lf}) -- re-read it")
    if r.get("pod_fp") != pf:
        return False, (f"ruled {r.get('date')} by {r.get('ruled_by')} about a different POD "
                       f"row ({r.get('pod_fp')}, now {pf}) -- re-read it")
    return True, None


def main():
    rows = load()
    if not rows:
        print(f"no rulings recorded ({os.path.relpath(PATH, ROOT)} absent or empty)")
        return 0
    print(f"{len(rows)} ruling(s) in {os.path.relpath(PATH, ROOT)}:\n")
    for led, key, r in rows:
        print(f"  {r.get('winner'):7s} {str(key):52s} {os.path.basename(led)}")
        print(f"          {r.get('why', '')[:104]}")
        print(f"          ruled_by {r.get('ruled_by')} {r.get('date')}  "
              f"local_fp {r.get('local_fp')} pod_fp {r.get('pod_fp')}")
    return 0


def _selftest():
    a = {"name": "n", "status": "ok", "result": "3.6%"}
    b = {"name": "n", "status": "fail", "result": ""}

    # Field ORDER must not change a fingerprint: these ledgers are written by several tools
    # and json.dumps preserves insertion order, so an order-sensitive hash would invalidate
    # every ruling the first time a different writer touched the row.
    reordered = {"result": "3.6%", "name": "n", "status": "ok"}
    assert fingerprint(a) == fingerprint(reordered), "fingerprint depends on field order"
    assert fingerprint(a) != fingerprint(b)
    assert fingerprint(None) is None

    # ANY edit invalidates, including to a field the reader never compared. A ruling is
    # about a pair of rows, not about the fields someone happened to look at.
    plus = dict(a, notes="added later")
    assert fingerprint(a) != fingerprint(plus), (
        "a field added after the ruling left the fingerprint unchanged, so the ruling would "
        "keep muting a key whose row has since been edited")

    idx = {("runs/experiments.jsonl", ("n", "t")): {
        "winner": "local", "ruled_by": "fb", "date": "2026-09-03",
        "local_fp": fingerprint(a), "pod_fp": fingerprint(b)}}
    ok, why = settled("runs/experiments.jsonl", ("n", "t"), a, b, idx)
    assert ok and why is None, (ok, why)

    # An unruled key is not settled, and says nothing -- it needs a decision, not a
    # re-decision.
    ok, why = settled("runs/experiments.jsonl", ("other", "t"), a, b, idx)
    assert not ok and why is None, (ok, why)

    # THE WHOLE POINT: a NEW row on either side reopens the key. Without this the ruling is
    # a permanent mute and the next disagreement under that key is silent.
    ok, why = settled("runs/experiments.jsonl", ("n", "t"), dict(a, result="9.9%"), b, idx)
    assert not ok and "LOCAL" in why, (ok, why)
    ok, why = settled("runs/experiments.jsonl", ("n", "t"), a, dict(b, result="now closed"), idx)
    assert not ok and "POD" in why, (ok, why)

    # A key list from JSON must compare equal to a keyfn's tuple, or every ruling read back
    # from disk misses and the file mutes nothing.
    ok, _ = settled("runs/experiments.jsonl", ["n", "t"], a, b, idx)
    assert ok, "a key read from JSON as a list did not match the tuple it was stored under"

    for led, key, r in load():
        assert r.get("winner") in WINNERS, f"{key}: winner {r.get('winner')!r}"
        for f in ("ledger", "key", "winner", "why", "ruled_by", "date", "local_fp", "pod_fp"):
            assert r.get(f) not in (None, ""), f"{key}: {f} is empty"
        assert led and led.startswith("runs/"), f"{key}: ledger {led!r}"

    print(f"ledger_resolutions selftest OK: fingerprint is order-insensitive and invalidated "
          f"by any field edit; a ruling settles a key only while BOTH rows are unchanged, "
          f"and a changed side reopens it naming which; an unruled key reports no reason; a "
          f"JSON list key matches its tuple; {len(load())} recorded ruling(s) well-formed")
    return 0


if __name__ == "__main__":
    sys.exit(_selftest() if "--selftest" in sys.argv else main())
