#!/usr/bin/env python3
"""Pod-only ledger rows come home (de-36). What happened only on the pod did not happen.

pod_push only ever pushes, and pod_drift only asserts that the files it LISTS match -- so a
row appended to runs/*.jsonl on the pod is invisible to every check in this repo. Five
score_matrix rows behind the closed A/Bs (3)/(2a)/(4) and b0-17's base lived only on the
pod's emptyDir until ce6ea53a moved them by hand.

MEASURED on 2026-09-03, which is the reason this is not a no-op: 2 pod-only score_matrix
keys (p500m_20b_0902 step1500 and step2500, the live run's own measurements) and 14
pod-only experiments rows. The task's reading predicted "0 missing after ce6ea53a"; that
was wrong, and the pod had kept accumulating.

    python3 scripts/pod_pull_ledgers.py            # report only, touches nothing
    python3 scripts/pod_pull_ledgers.py --apply    # append the missing rows locally
    python3 scripts/pod_pull_ledgers.py --selftest # the derivation, on known answers

IDENTITY COMES FROM ledger_audit.KEYS, not from a second copy here. That module already
decided each ledger's key at its writer, with the measurement for why (score_matrix is
(ckpt, profile) because write_records replaces on that pair; milestones is
(ckpt, milestone) after the first version keyed all 13 rows to None). A private key table
would drift from the audit's, and then two files would disagree about what a duplicate is.

LINE COUNTS CARRY NO INFORMATION HERE. milestones is 13 local against 10 on the pod and has
zero pod-only keys; score_matrix is 48 against 50 and has two. A count comparison would
have reported milestones as the urgent one and score_matrix as fine, both backwards.

WHAT IT REFUSES: a pod row whose key matches a local row with DIFFERENT content is not
appended. Two rows under one key have no defined current state, and for score_matrix
write_records raises on exactly that. A collision is a decision for a human -- which
measurement is right -- so it is reported and the row is left alone.

# restartable: reads the pod read-only; --apply appends locally through each writer.
"""

import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

POD_ROOT = "/work/aupai"
LEDGERS = ("runs/score_matrix.jsonl", "runs/experiments.jsonl",
           "runs/review.jsonl", "runs/milestones.jsonl")


def _keys():
    from ledger_audit import KEYS
    return KEYS


def read_pod(rel, pod_root=POD_ROOT):
    """(text, error). A MISSING file is an error, never an empty ledger.

    runs/review.jsonl does not exist on the pod today. Returning "" for it would report
    "0 pod-only rows", which reads as agreement when it means the question was not asked --
    the shape this repo has bought three times (unmeasured labelled as absent)."""
    p = f"{pod_root}/{rel}"
    r = subprocess.run([os.path.expanduser("~/bin/pod"), f"cat {p}"],
                       capture_output=True, text=True)
    if r.returncode != 0 or (not r.stdout and "No such file" in (r.stderr or "")):
        return None, f"unreadable on the pod ({(r.stderr or 'rc=%d' % r.returncode).strip()[:90]})"
    if not r.stdout.strip():
        return None, "empty or absent on the pod"
    return r.stdout, None


def parse(text):
    """(rows, n_unparseable). Unparseable lines are counted, never dropped silently."""
    rows, bad = [], 0
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            rows.append(json.loads(ln))
        except ValueError:
            bad += 1
    return rows, bad


def _empty(v):
    return v in (None, "", [], {}) or v == "no result recorded"


# A pod row still saying `running` while the local row is closed is the pod being BEHIND,
# not disagreeing: the run ended and only the repository heard. Measured: 59 of 155
# differing experiments rows differ on `status` alone, all of this shape (de-36,
# 2026-09-03). Without this, every one of them files as a contradiction and the 53 rows
# that actually need a human are buried in a list of 155.
_OPEN = {"running", ""}


def classify(local_row, pod_row):
    """Why a key present on BOTH sides differs. Returns 'stale' | 'result_only_on_pod' |
    'contradicts'.

    "The content differs" is not actionable and was the first version's whole predicate:
    it reported 155 collisions on the real ledgers, which is not 155 conflicts, it is two
    copies of a ledger at different points in each row's life. The classes had to be
    measured to find (de-36, 2026-09-03):

      stale               the pod adds nothing the local row does not already say: every
                          non-empty pod field agrees locally, OR the only disagreement is
                          a pod row still open where the local row is closed. 59 of 155
                          are that second form, and they are why this class needed the
                          `_OPEN` clause rather than plain field agreement.
      result_only_on_pod  the pod carries a non-empty `result` where the local row has
                          none. 53 of 155. A measurement the repository does not have,
                          which is R10 exactly, and the reason de-36 exists.
      contradicts         both sides state a different non-empty value. A human decides.

    Nothing in this class is ever applied. A local row is somebody's written decision --
    fb closed 62 of these as killed on 2026-09-01 -- and overwriting it from a stale pod
    copy would be an automated edit to a human's judgement in the direction of the older
    evidence."""
    pr, lr = pod_row.get("result"), local_row.get("result")
    disagree = [f for f, v in pod_row.items() if not _empty(v) and local_row.get(f) != v]
    if not disagree:
        return "stale"
    if disagree == ["status"] and pod_row.get("status") in _OPEN:
        return "stale"
    if not _empty(pr) and _empty(lr):
        return "result_only_on_pod"
    return "contradicts"


def diff_rows(pod_rows, local_rows, keyfn):
    """(missing, collisions). missing: pod rows whose key is absent locally. collisions:
    [(key, why, local_last, pod_last)] for a key on both sides whose CURRENT rows differ.

    BOTH SIDES FOLD TO THEIR LAST ROW BEFORE ANYTHING IS COMPARED. These ledgers are
    append-folds: exp.py done appends a closing row rather than rewriting the open one, so
    a key's current state is its last row and every earlier row under it is history.

    The first version folded only the LOCAL side and compared EVERY pod row against it. On
    the real ledgers that reported 161 differences, and the number was an artifact of the
    method: the pod holds up to five rows under one key (sft_p324_v3 has ok, running,
    killed, fail, killed), so one key alone contributed four "differences" against a local
    row that in fact matched its last one. Folded both sides: 171 of 185 shared keys AGREE
    and 14 differ. The 53 "the pod has a result and the repository does not" rows were the
    same artifact -- the pod's own last row for all 53 is the identical close the local row
    carries, because those closes were written ON the pod and pulled home. Measured
    2026-09-03: 178 pod rows are byte-identical to the current local last row.

    A report of 161 where the answer is 14 is not a loud version of the same finding. It
    made the pod look older than the repository when the pod is a superset of it, and a
    ruling issued on those numbers would have rewritten 53 correct local rows from rows
    the pod itself supersedes."""
    local, pod_last, order = {}, {}, []
    for r in local_rows:
        local[keyfn(r)] = r
    for r in pod_rows:
        k = keyfn(r)
        if k not in pod_last:
            order.append(k)
        pod_last[k] = r
    missing, collisions = [], []
    for k in order:
        r = pod_last[k]
        if k in local:
            if local[k] != r:
                collisions.append((k, classify(local[k], r), local[k], r))
        else:
            missing.append(r)
    return missing, collisions


def append_rows(rel, rows, root=ROOT):
    """Append through the ledger's own writer where it has one.

    score_matrix goes through eval.score_matrix.write_records so its duplicate guard and
    its flock run -- the task's requirement, and the reason this is not a plain append: that
    writer REPLACES same-(ckpt, profile) rows and raises when handed a key twice. It is
    given the union, because it rewrites the whole file from what it is passed.

    The other three are append-only ledgers whose writers take command-line arguments
    (exp.py start/done, harness task) rather than a row, so a row already formed cannot be
    passed through them; appending the line is what those writers do."""
    path = os.path.join(root, rel)
    if rel == "runs/score_matrix.jsonl":
        sys.path.insert(0, root)
        from eval.score_matrix import write_records
        existing, _ = parse(open(path, encoding="utf-8").read()) if os.path.exists(path) else ([], 0)
        write_records(path, existing + rows)
        return len(rows)
    with open(path, "a", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def survey(root=ROOT, pod_root=POD_ROOT, reader=read_pod):
    """[(rel, n_pod, n_local, missing, collisions, error)] for every ledger.

    Collisions a recorded ruling settles are dropped -- see scripts/ledger_resolutions.py.
    A ruling holds only while both sides still carry the rows it was made about; when one
    changes, the key comes back as a collision whose `why` says the ruling was about other
    rows, so a settled key stays quiet and a NEW disagreement under it does not."""
    from ledger_resolutions import index, settled

    keys = _keys()
    idx = index()
    out = []
    for rel in LEDGERS:
        text, err = reader(rel, pod_root)
        if err:
            out.append((rel, 0, 0, [], [], err))
            continue
        pod_rows, bad = parse(text)
        lpath = os.path.join(root, rel)
        local_rows, _ = parse(open(lpath, encoding="utf-8").read()) if os.path.exists(lpath) else ([], 0)
        missing, coll = diff_rows(pod_rows, local_rows, keys[rel])
        kept = []
        for k, why, lrow, prow in coll:
            ok, stale = settled(rel, k, lrow, prow, idx)
            if ok:
                continue
            kept.append((k, stale or why, lrow, prow))
        note = f"{bad} unparseable pod line(s)" if bad else None
        n_ruled = len(coll) - len(kept)
        if n_ruled:
            note = f"{note}; {n_ruled} settled by a ruling" if note else f"{n_ruled} settled by a ruling"
        out.append((rel, len(pod_rows), len(local_rows), missing, kept, note))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="append the missing rows locally (default: report only)")
    ap.add_argument("--pod-root", default=POD_ROOT)
    a = ap.parse_args(argv)

    rows = survey(pod_root=a.pod_root)
    total_missing = 0
    by_class = {}
    for rel, npod, nloc, missing, coll, note in rows:
        total_missing += len(missing)
        cls = {}
        for _k, why, _l, _p in coll:
            cls[why] = cls.get(why, 0) + 1
            by_class[why] = by_class.get(why, 0) + 1
        tag = f"  [{note}]" if note else ""
        summary = ", ".join(f"{w} {n}" for w, n in sorted(cls.items())) or "none"
        print(f"{rel:28s} pod {npod:>4} local {nloc:>4}  pod-only {len(missing):>3}  "
              f"differing: {summary}{tag}")
        for k, why, lrow, prow in [c for c in coll if c[1] == "result_only_on_pod"][:3]:
            print(f"    RESULT ONLY ON POD {k}")
            print(f"      pod   [{prow.get('status')}] {str(prow.get('result'))[:88]}")
            print(f"      local [{lrow.get('status')}] {str(lrow.get('result'))[:88]}")
    print()
    n_res = by_class.get("result_only_on_pod", 0)
    if n_res:
        print(f"{n_res} row(s) where THE POD HOLDS A RESULT AND THE REPOSITORY DOES NOT. This "
              f"is R10 in the ledger itself: the measurement happened, and no local artifact "
              f"records it. NOT applied -- a local row is a written decision (fb closed a "
              f"batch of these as killed on 2026-09-01) and overwriting it from an older pod "
              f"copy would automate an edit to a human's judgement. Read them and decide.")
    if by_class.get("contradicts"):
        print(f"{by_class['contradicts']} row(s) where both sides state a different non-empty "
              f"value. A human decides which measurement is right.")
    if not total_missing:
        print("no pod-only rows: every pod row's key is present locally.")
        return 0
    if not a.apply:
        print(f"{total_missing} pod-only row(s). Re-run with --apply to append them.")
        return 0
    for rel, _npod, _nloc, missing, _coll, _note in rows:
        if missing:
            n = append_rows(rel, missing)
            print(f"appended {n} row(s) to {rel}")
    return 0


def _selftest():
    """Known answers on fabricated pod content, plus the real key table.

    Every case is a world where the answer is known by construction, because the live
    numbers change every day and an assertion against them would be a permanent red as
    soon as someone pulled the rows (de-35, this morning)."""
    keys = _keys()
    for rel in LEDGERS:
        assert rel in keys, f"{rel} has no key in ledger_audit.KEYS"

    kf = keys["runs/score_matrix.jsonl"]
    a = {"ckpt": "c1", "profile": "full", "v": 1}
    b = {"ckpt": "c2", "profile": "full", "v": 2}
    b_changed = {"ckpt": "c2", "profile": "full", "v": 99}

    missing, coll = diff_rows([a, b], [a], kf)
    assert [r["ckpt"] for r in missing] == ["c2"], missing
    assert not coll, coll

    # A pod row whose key exists locally with DIFFERENT content is not a missing row:
    # appending it would put two rows under one key, which write_records raises on.
    missing, coll = diff_rows([b_changed], [b], kf)
    assert not missing, f"a colliding row must not be offered for append: {missing}"
    assert len(coll) == 1 and coll[0][0] == ("c2", "full"), coll

    # THE THREE CLASSES, which the first version collapsed into "the content differs" and
    # reported 155 of on the real ledgers -- a number that named nothing to act on.
    stale = {"ckpt": "c3", "profile": "full", "result": "", "status": "running"}
    closed = {"ckpt": "c3", "profile": "full", "result": "", "status": "killed",
              "finding": "closed by fb"}
    assert classify(closed, stale) == "stale", classify(closed, stale)
    pod_has = {"ckpt": "c4", "profile": "full", "result": "code-500 40.0%", "status": "ok"}
    loc_none = {"ckpt": "c4", "profile": "full", "result": "no result recorded",
                "status": "killed"}
    assert classify(loc_none, pod_has) == "result_only_on_pod", classify(loc_none, pod_has)
    # "no result recorded" is the phrase fb's batch close wrote, and treating it as a value
    # rather than as absence hides all 53 of these in the contradicts bucket.
    assert _empty("no result recorded")
    both = {"ckpt": "c5", "profile": "full", "result": "40.0%"}
    other = {"ckpt": "c5", "profile": "full", "result": "2.2%"}
    assert classify(both, other) == "contradicts", classify(both, other)

    # THE LAST local row under a key is current, not the first: exp.py done APPENDS a
    # closing row. With setdefault, a closed run reads as still open and its close is
    # invisible to every class above.
    ek = keys["runs/experiments.jsonl"]
    open_row = {"name": "n", "started": "t", "status": "running", "result": ""}
    close_row = {"name": "n", "started": "t", "status": "ok", "result": "3.6%"}
    missing, coll = diff_rows([open_row], [open_row, close_row], ek)
    assert not missing and len(coll) == 1, (missing, coll)
    assert coll[0][2]["status"] == "ok", (
        "diff_rows compared against the FIRST local row under the key; exp.py done appends "
        "a close, so the last row is the current one")

    # BOTH SIDES FOLD TO THEIR LAST ROW. The pod holds up to five rows under one key
    # (sft_p324_v3: ok, running, killed, fail, killed), and comparing each of them against
    # the local row is what produced 161 differences where the answer is 14.
    missing, coll = diff_rows([b, b_changed], [a], kf)
    assert len(missing) == 1, f"a key repeated on the pod was offered twice: {missing}"
    assert missing[0]["v"] == 99, (
        f"the FIRST pod row under a repeated key was kept, got v={missing[0]['v']}. The pod "
        f"holds a run's open row and its close under one key; keeping the first writes the "
        f"open row and abandons the result")
    # The property the first version got wrong, as a known answer: a pod key whose EARLIER
    # rows differ but whose LAST row matches locally is agreement, not a collision.
    missing, coll = diff_rows([{"ckpt": "c1", "profile": "full", "v": 7}, a], [a], kf)
    assert not missing and not coll, (
        f"an earlier pod row under a key whose LAST row matches locally was reported as a "
        f"difference: {coll}. That is the artifact that turned 14 into 161 -- the pod is an "
        f"append log, and every row but the last is history")

    # An identical row is neither missing nor a collision.
    missing, coll = diff_rows([a], [a], kf)
    assert not missing and not coll, (missing, coll)

    # A MISSING pod file must not read as an empty ledger. runs/review.jsonl does not
    # exist on the pod, and "0 pod-only rows" for it would report agreement where the
    # question was never asked.
    def _no_files(rel, pod_root):
        return None, "empty or absent on the pod"
    rows = survey(reader=_no_files)
    assert len(rows) == len(LEDGERS)
    assert all(r[5] for r in rows), f"a missing pod file reported no error: {rows}"
    assert all(not r[3] for r in rows), "a missing file must offer no rows to append"

    # LINE COUNTS ARE NOT THE ANSWER, and this is the property that decides whether the
    # key comparison was needed at all. Real 2026-09-03 shape: milestones is 13 local
    # against 10 pod with ZERO pod-only keys, while score_matrix is 48 against 50 with
    # two. A count test calls the first urgent and the second clean, both backwards.
    m = keys["runs/milestones.jsonl"]
    big_local = [{"ckpt": f"c{i}", "milestone": "m"} for i in range(13)]
    small_pod = [{"ckpt": f"c{i}", "milestone": "m"} for i in range(10)]
    missing, _ = diff_rows(small_pod, big_local, m)
    assert not missing, ("a pod file with FEWER rows can still be a subset -- if this "
                         "reports rows, the comparison is counting, not keying")

    print("pod_pull_ledgers selftest OK: 4 ledgers keyed from ledger_audit.KEYS; "
          "missing/duplicate-key/identical on known answers; all three difference classes "
          "(stale, result_only_on_pod, contradicts); BOTH sides fold to their last row, "
          "asserted on the case that produced 161 differences where the answer is 14 -- an "
          "earlier pod row under a key whose last row matches locally; a missing pod file "
          "reports an error instead of zero; and a pod file with FEWER rows correctly "
          "yields nothing to pull")
    return 0


if __name__ == "__main__":
    sys.exit(_selftest() if "--selftest" in sys.argv else main())
