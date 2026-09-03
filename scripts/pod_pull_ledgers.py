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
    python3 scripts/pod_pull_ledgers.py --push     # report what the POD lacks (de-41)
    python3 scripts/pod_pull_ledgers.py --push --apply   # append those rows ON the pod
    python3 scripts/pod_pull_ledgers.py --selftest # the derivation, on known answers

THE REVERSE DIRECTION EXISTS TOO, AND IT HAD NO APPROVED PATH (de-41). pod_push.sh --all
excludes runs/ from both its push list and its dirty count, deliberately -- the pod writes
those rows and a whole-file overwrite would destroy whatever it wrote since the last sync.
The consequence was that a close written on main could never reach the pod: three rows
(p500m_20b_0902, control_lr_scan x2) were closed here and still read `running` there, and
no_ghost_running's authority IS the pod, so each turned the launch gate red for the next
launcher and each needed a human to hand-copy the close. The gate was right every time --
the pod really did hold a running row with no process -- so the defect was the missing
direction, not the check.

--push is that direction and it is the SAME fold, classification and ruling logic with the
sides swapped, not a second implementation. It appends; it never rewrites a pod file, so a
row the pod wrote after the last sync cannot be lost. What it refuses is what --apply
refuses, in the mirror: a local row whose key exists on the pod with different content is a
collision for a human, not a row to append.

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
LEDGERS = ("runs/score_matrix.jsonl", "runs/experiments.jsonl", "runs/review.jsonl", "runs/milestones.jsonl")


def _keys():
    from ledger_audit import KEYS

    return KEYS


def read_pod(rel, pod_root=POD_ROOT):
    """(text, error). A MISSING file is an error, never an empty ledger.

    runs/review.jsonl does not exist on the pod today. Returning "" for it would report
    "0 pod-only rows", which reads as agreement when it means the question was not asked --
    the shape this repo has bought three times (unmeasured labelled as absent)."""
    p = f"{pod_root}/{rel}"
    r = subprocess.run([os.path.expanduser("~/bin/pod"), f"cat {p}"], capture_output=True, text=True)
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

# Fields that are PROVENANCE, not measurement. A key differing only on these is not two
# measurements disagreeing, and calling it `contradicts` puts it in the class whose text says
# "a human decides which measurement is right" -- there is no measurement in dispute.
#
# MEASURED 2026-09-03 on the real ledgers under --push: 6 of 7 reported contradictions were
# this, in three shapes, and every one is a repair the repository already made:
#   commit  local e993143 vs pod cec145b -- and `cec145b` names no object here, because
#           exp.py wrote abbreviated shas that auto-scale with the object count until de-38.
#           The local row IS the corrected one.
#   notes   local "commit was 10654b1, which names no object in this repository" vs pod "",
#           with commit local `unknown` vs pod `10654b1`. Same repair, one step further: the
#           unresolvable sha was replaced by the placeholder and the reason recorded in notes.
#           Reported as a contradiction, this reads as the repository having LOST the sha.
#   ended   local 02:32 vs pod 03:55 -- one side's monitor closed the row later. The result
#           and status agree to the character.
# Leaving these in `contradicts` would have buried the one real disagreement under six
# repairs, which is de-36's 161-vs-14 shape again in a different field.
_PROVENANCE = {"commit", "notes", "ended", "cmd", "socket", "reviewer"}


def classify(local_row, pod_row):
    """Why a key present on BOTH sides differs. Returns 'stale' | 'result_only_on_pod' |
    'provenance_only' | 'contradicts'.

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
    # Provenance-only, checked against BOTH sides' differing fields rather than the pod's
    # alone: local `notes` explaining an unresolvable sha is absent from the pod entirely, so
    # a pod-side scan cannot see it and the pair would still read as a contradiction.
    both = {f for f in set(local_row) | set(pod_row) if local_row.get(f) != pod_row.get(f)}
    if both and both <= _PROVENANCE:
        return "provenance_only"
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


def append_pod_rows(rel, rows, pod_root=POD_ROOT):
    """Append rows to a ledger ON THE POD (de-41). Returns (n, error).

    APPEND, NEVER A FILE PUSH. pod_push.sh --all excludes runs/ on purpose: the pod writes
    those rows, so overwriting the file would destroy every row appended there since the last
    sync. A shell append (`cat >>`) adds without reading, so the pod's own rows survive by
    construction.

    Content goes through base64 rather than into the command line. A result string carries
    quotes, percent signs and non-ASCII, and every one of those is shell syntax inside
    `pod "..."`; the row would arrive mangled or the command would not parse. Newlines are
    stripped from the encoding because a newline inside the quoted command becomes a command
    separator in the pod's bash -lc (the same trap pod_push.sh:184 documents for paths).

    The size cap is podput's, ~100KB of argv, and it is checked rather than assumed: a batch
    of closes is a few hundred bytes each, so the realistic call is far under it, but a silent
    truncation of a ledger write is exactly the failure this repo keeps buying."""
    import base64

    payload = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    b64 = base64.b64encode(payload.encode("utf-8")).decode("ascii")
    if len(b64) > 90000:
        return 0, (
            f"{len(b64)} base64 chars exceeds the ~100KB argv cap -- split the batch "
            f"(this is podput's limit, not a constant here)"
        )
    cmd = f"cd {pod_root} && printf %s {b64} | base64 -d >> {rel}"
    r = subprocess.run(
        [os.path.expanduser("~/bin/pod"), cmd], capture_output=True, text=True, stdin=subprocess.DEVNULL
    )
    if r.returncode:
        return 0, f"pod append failed: {(r.stderr or 'rc=%d' % r.returncode).strip()[:120]}"
    return len(rows), None


def verify_pod_append(rel, rows, pod_root=POD_ROOT, reader=read_pod):
    """Read the pod back and confirm each row's key now folds to the row we sent.

    A write whose success is inferred from rc=0 is not a verified write: the shell pipeline
    above can exit 0 with a truncated payload if base64 -d hits a short read, and a ledger
    that is silently short reads exactly like one that is correct. So this re-reads and folds
    -- the same fold every comparison here uses -- and reports which keys did not land."""
    keys = _keys()
    text, err = reader(rel, pod_root)
    if err:
        return [f"cannot verify: {err}"]
    pod_rows, _ = parse(text)
    kf = keys[rel]
    last = {}
    for r in pod_rows:
        last[kf(r)] = r
    bad = []
    for r in rows:
        k = kf(r)
        if k not in last:
            bad.append(f"{k}: absent after the append")
        elif last[k] != r:
            bad.append(f"{k}: pod's last row is not the one sent")
    return bad


def survey(root=ROOT, pod_root=POD_ROOT, reader=read_pod, push=False):
    """[(rel, n_pod, n_local, missing, collisions, error)] for every ledger.

    `push` swaps which side is authoritative for `missing`: by default missing is pod rows
    absent locally (pull); with push it is LOCAL rows absent on the pod (de-41). Collisions
    are the same set either way -- a key whose two current rows differ is one disagreement,
    not a directional one -- and classify() keeps its argument order, so `stale` continues to
    mean "the pod adds nothing" rather than reversing meaning under --push.

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
        if push:
            missing, coll = diff_rows(local_rows, pod_rows, keys[rel])
            # RESTORE (local, pod) AND RECLASSIFY. Swapping diff_rows' arguments swaps what
            # it hands classify, and classify is NOT symmetric: with the sides reversed, a
            # local close against a pod row still `running` came back as
            # `result_only_on_pod` -- the pod holding a result the repository lacks, which is
            # the one class that reads as "the local row is the stale one". That is exactly
            # backwards for a close, and it is the reading under which someone discards it.
            # Caught by the selftest case below, not by reading this.
            coll = [(k, classify(lrow, prow), lrow, prow) for k, _why, prow, lrow in coll]
        else:
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
    ap.add_argument("--apply", action="store_true", help="append the missing rows (default: report only)")
    ap.add_argument(
        "--push", action="store_true", help="the reverse direction: report/append rows the POD lacks (de-41)"
    )
    ap.add_argument("--pod-root", default=POD_ROOT)
    a = ap.parse_args(argv)

    rows = survey(pod_root=a.pod_root, push=a.push)
    side = "local-only" if a.push else "pod-only"
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
        print(f"{rel:28s} pod {npod:>4} local {nloc:>4}  {side} {len(missing):>3}  differing: {summary}{tag}")
        for k, why, lrow, prow in [c for c in coll if c[1] == "result_only_on_pod"][:3]:
            print(f"    RESULT ONLY ON POD {k}")
            print(f"      pod   [{prow.get('status')}] {str(prow.get('result'))[:88]}")
            print(f"      local [{lrow.get('status')}] {str(lrow.get('result'))[:88]}")
    print()
    n_res = by_class.get("result_only_on_pod", 0)
    if n_res:
        print(
            f"{n_res} row(s) where THE POD HOLDS A RESULT AND THE REPOSITORY DOES NOT. This "
            f"is R10 in the ledger itself: the measurement happened, and no local artifact "
            f"records it. NOT applied -- a local row is a written decision (fb closed a "
            f"batch of these as killed on 2026-09-01) and overwriting it from an older pod "
            f"copy would automate an edit to a human's judgement. Read them and decide."
        )
    if by_class.get("contradicts"):
        print(
            f"{by_class['contradicts']} row(s) where both sides state a different non-empty "
            f"value. A human decides which measurement is right."
        )
    if not total_missing:
        print(f"no {side} rows: every key on one side is present on the other.")
        return 0
    if not a.apply:
        print(f"{total_missing} {side} row(s). Re-run with --apply to append them.")
        return 0
    rc = 0
    for rel, _npod, _nloc, missing, _coll, _note in rows:
        if not missing:
            continue
        if a.push:
            n, err = append_pod_rows(rel, missing, a.pod_root)
            if err:
                print(f"REFUSING {rel}: {err}")
                rc = 1
                continue
            bad = verify_pod_append(rel, missing, a.pod_root)
            if bad:
                print(f"APPENDED {n} row(s) to {rel} ON THE POD but VERIFY FAILED:")
                for b in bad:
                    print(f"    {b}")
                rc = 1
            else:
                print(f"appended {n} row(s) to {rel} on the pod (verified by re-read)")
        else:
            n = append_rows(rel, missing)
            print(f"appended {n} row(s) to {rel}")
    return rc


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
    closed = {"ckpt": "c3", "profile": "full", "result": "", "status": "killed", "finding": "closed by fb"}
    assert classify(closed, stale) == "stale", classify(closed, stale)
    pod_has = {"ckpt": "c4", "profile": "full", "result": "code-500 40.0%", "status": "ok"}
    loc_none = {"ckpt": "c4", "profile": "full", "result": "no result recorded", "status": "killed"}
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
        "a close, so the last row is the current one"
    )

    # BOTH SIDES FOLD TO THEIR LAST ROW. The pod holds up to five rows under one key
    # (sft_p324_v3: ok, running, killed, fail, killed), and comparing each of them against
    # the local row is what produced 161 differences where the answer is 14.
    missing, coll = diff_rows([b, b_changed], [a], kf)
    assert len(missing) == 1, f"a key repeated on the pod was offered twice: {missing}"
    assert missing[0]["v"] == 99, (
        f"the FIRST pod row under a repeated key was kept, got v={missing[0]['v']}. The pod "
        f"holds a run's open row and its close under one key; keeping the first writes the "
        f"open row and abandons the result"
    )
    # The property the first version got wrong, as a known answer: a pod key whose EARLIER
    # rows differ but whose LAST row matches locally is agreement, not a collision.
    missing, coll = diff_rows([{"ckpt": "c1", "profile": "full", "v": 7}, a], [a], kf)
    assert not missing and not coll, (
        f"an earlier pod row under a key whose LAST row matches locally was reported as a "
        f"difference: {coll}. That is the artifact that turned 14 into 161 -- the pod is an "
        f"append log, and every row but the last is history"
    )

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
    assert not missing, (
        "a pod file with FEWER rows can still be a subset -- if this "
        "reports rows, the comparison is counting, not keying"
    )

    # --- THE REVERSE DIRECTION (de-41). A mirror implementation fails at the reversal, so
    # every case below is about the swap itself rather than about the fold again.
    pod_only = {"name": "p", "started": "t", "status": "ok", "result": "on the pod only"}
    loc_only = {"name": "l", "started": "t", "status": "ok", "result": "here only"}

    def _one(rel, pod_root, _pod=[pod_only]):
        return "".join(json.dumps(r) + "\n" for r in _pod), None

    # pull: the pod-only row is offered; push: the LOCAL-only row is, and they are different
    # rows. A survey that returned the same `missing` under both flags would be the bug.
    import tempfile

    d = tempfile.mkdtemp()
    os.makedirs(os.path.join(d, "runs"))
    for rel in LEDGERS:
        with open(os.path.join(d, rel), "w", encoding="utf-8") as fh:
            fh.write(json.dumps(loc_only) + "\n")
    try:
        pulled = survey(root=d, reader=_one, push=False)
        pushed = survey(root=d, reader=_one, push=True)
    finally:
        import shutil

        shutil.rmtree(d, ignore_errors=True)
    pull_exp = [r for r in pulled if r[0] == "runs/experiments.jsonl"][0]
    push_exp = [r for r in pushed if r[0] == "runs/experiments.jsonl"][0]
    assert [r["name"] for r in pull_exp[3]] == ["p"], f"pull must offer the POD-only row, got {pull_exp[3]}"
    assert [r["name"] for r in push_exp[3]] == ["l"], (
        f"push must offer the LOCAL-only row, got {push_exp[3]}. If this equals the pull "
        f"result, --push is reporting the same direction under a different label"
    )
    assert pull_exp[3] != push_exp[3], "the two directions returned the same rows"
    # n_pod and n_local keep their meaning under --push: the header must not swap them too,
    # or the printed line says the pod has the local count.
    assert push_exp[1] == 1 and push_exp[2] == 1, (push_exp[1], push_exp[2])

    # CLASSIFY KEEPS ITS ARGUMENT ORDER UNDER --push. `stale` means "the pod adds nothing";
    # if the swap reached classify, a pod row still `running` against a local close would
    # come back as the local side being stale, i.e. the exact reading that makes a close
    # look like the thing to discard.
    def _stale_pod(rel, pod_root):
        return json.dumps({"name": "n", "started": "t", "status": "running", "result": ""}) + "\n", None

    d2 = tempfile.mkdtemp()
    os.makedirs(os.path.join(d2, "runs"))
    for rel in LEDGERS:
        with open(os.path.join(d2, rel), "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"name": "n", "started": "t", "status": "ok", "result": "3.6%"}) + "\n")
    try:
        pushed2 = survey(root=d2, reader=_stale_pod, push=True)
    finally:
        import shutil as _sh

        _sh.rmtree(d2, ignore_errors=True)
    exp2 = [r for r in pushed2 if r[0] == "runs/experiments.jsonl"][0]
    assert not exp2[3], f"a key present on BOTH sides is not a row to append: {exp2[3]}"
    assert len(exp2[4]) == 1 and exp2[4][0][1] == "stale", (
        f"under --push, a pod row still `running` against a local close must classify as "
        f"stale (the pod adds nothing), got {exp2[4]}"
    )
    assert exp2[4][0][2]["status"] == "ok" and exp2[4][0][3]["status"] == "running", (
        f"the collision tuple's (local, pod) order was not restored after the swap: "
        f"{exp2[4][0][2]}, {exp2[4][0][3]}"
    )

    # THE APPEND IS SIZE-CHECKED, not assumed. A silent truncation of a ledger write is the
    # failure this repo keeps buying, so the cap refuses rather than sending a short payload.
    big = [{"name": "x" * 70000, "started": "t"}]
    n, err = append_pod_rows("runs/experiments.jsonl", big, pod_root="/nonexistent")
    assert n == 0 and err and "argv cap" in err, (n, err)

    # PROVENANCE-ONLY, from the three real shapes --push found on 2026-09-03. Each pair agrees
    # on status AND result to the character, so none is two measurements disagreeing, and
    # calling them `contradicts` put six repairs in front of the one real disagreement.
    base = {"name": "n", "started": "t", "status": "ok", "result": "3.6%"}
    # (1) an abbreviated sha the repository corrected (de-38: --short auto-scales)
    assert classify({**base, "commit": "e993143"}, {**base, "commit": "cec145b"}) == "provenance_only"
    # (2) the same repair one step on: sha replaced by a placeholder, reason in notes. The
    # local `notes` is ABSENT from the pod row, so a pod-side field scan cannot see it -- this
    # case is why the check reads both sides' differing fields.
    assert (
        classify(
            {**base, "commit": "unknown", "notes": "commit was 10654b1, which names no object"},
            {**base, "commit": "10654b1", "notes": ""},
        )
        == "provenance_only"
    )
    # (3) two monitors closed the same row at different times
    assert classify({**base, "ended": "02:32"}, {**base, "ended": "03:55"}) == "provenance_only"
    # AND THE NEGATIVE, which is what makes the class a class: a differing RESULT stays a
    # contradiction even when a provenance field differs too. Without this the allow-list
    # could grow until it swallowed the real disagreements.
    assert (
        classify(
            {**base, "result": "3.6%", "commit": "aaa"},
            {**base, "result": "9.9%", "commit": "bbb"},
        )
        == "contradicts"
    )
    assert classify({**base, "status": "ok"}, {**base, "status": "fail"}) == "contradicts"

    print(
        "pod_pull_ledgers selftest OK: 4 ledgers keyed from ledger_audit.KEYS; "
        "missing/duplicate-key/identical on known answers; all three difference classes "
        "(stale, result_only_on_pod, contradicts) plus provenance_only on its three real shapes -- corrected sha, sha-to-placeholder with the reason in notes, two monitors closing at different times -- with the negative that a differing result or status stays a contradiction; BOTH sides fold to their last row, "
        "asserted on the case that produced 161 differences where the answer is 14 -- an "
        "earlier pod row under a key whose last row matches locally; a missing pod file "
        "reports an error instead of zero; and a pod file with FEWER rows correctly "
        "yields nothing to pull. REVERSE DIRECTION (de-41): --push offers the local-only "
        "row where --pull offers the pod-only one and the two sets are asserted different; "
        "n_pod/n_local keep their meaning; classify keeps its (local, pod) order under the "
        "swap -- the case that caught the live bug, where a local close against a stale pod "
        "row classified as result_only_on_pod, the one reading under which a close looks "
        "like the row to discard; and the append refuses an over-cap payload rather than "
        "truncating it"
    )
    return 0


if __name__ == "__main__":
    sys.exit(_selftest() if "--selftest" in sys.argv else main())
