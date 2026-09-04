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


def _ledgers():
    """Every ledger with a measured row identity -- derived from ledger_audit.KEYS, not listed here.

    DL-11 (S1, audit 2026-09-04) IS THIS FUNCTION'S REASON. The hand-written tuple that stood here
    named four files, and the audit measured what that cost: `tasks.jsonl` 21 local-only ids,
    `board.jsonl` 45 local-only keys, `friction.jsonl` 37 local-only, `review.jsonl` 167 local rows
    against a file that does not exist on the pod, `artifact_refs.jsonl` 21 pod-only. Four ledgers
    had no transport in EITHER direction, and the divergence was structural rather than accidental:
    pod_push.sh excludes runs/, so a list that omits a ledger omits it permanently.

    The docstring above already argued that identity must come from KEYS rather than a second copy,
    for exactly this failure mode -- and then the file kept a second copy of WHICH LEDGERS EXIST,
    one table lower. A ledger gains a key in ledger_audit and this transport still cannot see it.

    Sorted for a stable report order. `artifact_refs.jsonl` is still absent because it has no
    KEYS entry (DL-12: two schemas, no row identity) -- and it is absent by that fact rather than by
    a decision here, so the day it gets a key it gets transport too.
    """
    return tuple(sorted(_keys()))


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
_OPEN = {"running", "open", ""}

# THE SAME PROPERTY, IN THE FIELD EACH LEDGER NAMES IT (C4/DL-11, de, 2026-09-04). experiments
# rows carry `status: running`; tasks rows carry `state: open`. The clause below knew only the
# first, so widening the transport to tasks.jsonl reported 53 contradictions -- and MEASURED, 39 of
# them differ on `state` ALONE and every one is local-closed against pod-open, i.e. exactly the
# staleness this clause exists to name. Reporting 53 rows as needing a human when 39 need nothing
# is de-36's 161-vs-14 shape a third time, in a third field.
_OPEN_FIELDS = ("status", "state")

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
    disagree = [f for f, v in pod_row.items() if not _empty(v) and local_row.get(f) != v]
    if not disagree:
        return "stale"
    if len(disagree) == 1 and disagree[0] in _OPEN_FIELDS and pod_row.get(disagree[0]) in _OPEN:
        return "stale"
    # Provenance-only, checked against BOTH sides' differing fields rather than the pod's
    # alone: local `notes` explaining an unresolvable sha is absent from the pod entirely, so
    # a pod-side scan cannot see it and the pair would still read as a contradiction.
    both = {f for f in set(local_row) | set(pod_row) if local_row.get(f) != pod_row.get(f)}
    if both and both <= _PROVENANCE:
        return "provenance_only"
    # WHICH FIELD CARRIES THE CONTENT depends on the ledger, and hardcoding `result` made this
    # class blind on tasks.jsonl. Measured 2026-09-04 under --push with the widened list: 14 task
    # rows have a `reading` on the pod and NONE locally, and in 0 of them do both sides carry a
    # different non-empty value -- so all 14 are content the repository does not have, which is
    # this class exactly, and every one was reported as `contradicts`: filed as "a human decides
    # which is right" when there is nothing to decide, only something to copy.
    _content = [f for f in ("result", "reading", "evidence")
                if not _empty(pod_row.get(f)) and _empty(local_row.get(f))]
    _contested = [f for f in ("result", "reading", "evidence")
                  if not _empty(pod_row.get(f)) and not _empty(local_row.get(f))
                  and pod_row.get(f) != local_row.get(f)]
    if _content and not _contested:
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

    THE OTHER EIGHT ARE APPENDED AS LINES, and that is safe for a reason worth stating rather
    than assuming, because two of them are rewrite-style. Their writers take command-line
    arguments (exp.py start/done, harness task/friction) rather than a row, so a row already
    formed cannot be passed through them -- and appending is what those writers do at the
    syscall level anyway (harness.py:8114, O_APPEND).

    The rewrite-style pair needs the argument spelled out. tasks.jsonl is written BOTH ways:
    `harness task` appends an event (harness.py:8106, "Append, never rewrite"), while the
    pipeline paths fold with _read_tasks and rewrite with _write_tasks. Appending a row whose
    id already exists would therefore fold to the APPENDED row winning -- last row per id --
    which for a task closed here and still open on the pod would silently reopen it. That
    cannot happen: only rows whose KEY is absent on the receiving side are ever passed here.
    diff_rows classifies a key present on both sides as a collision, and no collision is ever
    applied. The same argument covers ledger_resolutions.jsonl.
    """
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


def events_pod_lacks(pod_rows, local_rows, keyfn):
    """Local EVENTS the pod does not have, per key, in local order (6e, 2026-09-04).

    THE ROW-LEVEL PUSH COULD NEVER SEND A CLOSE, which is the one thing it was built for.
    diff_rows compares each side's LAST row per key, so a key present on both sides is never
    `missing` -- and a close is not a new key, it is a second event under an existing one. The
    three closed-but-running rows de-41's own text names were therefore unreachable by
    --push: classify() correctly calls them `stale` (the pod adds nothing), collisions are
    never applied, and `missing` is keyed. Every branch behaved as designed and the case fell
    between them.

    Identity here is the EVENT, not the key: (key, status, result, ended). A close and its
    start share a key and differ in status, so the pod lacking the close is visible only at
    this granularity.

    ONLY EVENTS AFTER THE POD'S LAST ONE FOR THAT KEY. Sending an older `running` start after
    the pod already holds an `ok` folds the row back to running -- exp.fold's terminal-wins
    rule protects against a later start, but the pod's file order would then put a start last
    and any reader folding on position reads it as open. 6e hit this by hand and left two such
    events out. "Later" is the event's own position in the local file: the local ledger is an
    append log, so its order IS the sequence, and timestamps are not comparable across
    machines (`ended` is absent on a start row entirely).

    Reserved for a human: two CLOSES under one key with different values. That is a genuine
    disagreement about a measurement, and it stays in classify()'s `contradicts`.
    """
    def sig(r):
        return (keyfn(r), str(r.get("status", "")), str(r.get("result", "")),
                str(r.get("ended", "")))

    pod_sigs = {sig(r) for r in pod_rows}
    # The pod's last event per key, by its own file order: what "later than" is measured from.
    pod_last_idx = {}
    for i, r in enumerate(pod_rows):
        pod_last_idx[keyfn(r)] = i
    # The local index of the event matching the pod's last one for that key. Anything before
    # it is history the pod has already moved past.
    local_cut = {}
    for i, r in enumerate(local_rows):
        k = keyfn(r)
        if k in pod_last_idx and sig(r) == sig(pod_rows[pod_last_idx[k]]):
            local_cut[k] = i
    out = []
    for i, r in enumerate(local_rows):
        k = keyfn(r)
        if sig(r) in pod_sigs:
            continue
        cut = local_cut.get(k)
        if cut is None:
            # TWO CASES, ONE ANSWER, and the answer is refuse. Either the pod lacks this key
            # entirely -- diff_rows' `missing` already offers it, and returning it here would
            # append the same row twice -- or the pod's current event for the key is not in
            # the local file at all, so local order cannot say whether this event precedes or
            # follows it, and sending it might fold the row backwards.
            #
            # A separate `k not in pod_last_idx` guard stood above this and was REMOVED as
            # dead: measured 2026-09-04, removing it changes no output, because a key the pod
            # lacks has no cut either. It read as a second safeguard and no test could tell it
            # from its absence, which is worse than one check that is known to cover both.
            continue
        if i <= cut:
            continue
        out.append(r)
    return out


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

    The size cap is podput's, ~100KB of argv. THE FIRST VERSION REFUSED AN OVER-CAP BATCH and its
    docstring said "a batch of closes is a few hundred bytes each, so the realistic call is far
    under it" -- measured wrong on the first real use: C4's tasks.jsonl push is 52 rows carrying
    `reading`, `why` and `evidence` prose and encodes to 120,108 base64 chars. The refusal was still
    the right behaviour over a silent truncation, but refusing the whole batch means the transport
    cannot move the one ledger that most needed it.

    So it CHUNKS, and each chunk is a complete append of whole rows -- never a split row, because
    half a JSON line in an append-only ledger is a parse error for every reader afterwards. A chunk
    failing stops the rest and reports how many landed: a partial append is recoverable (the next
    run sees the remainder as still missing and sends it), while a wrong count is not."""
    import base64

    if not rows:
        return 0, None
    # One row at a time into the current chunk, so the boundary always falls between rows. A row
    # that alone exceeds the cap is reported rather than sent truncated.
    #
    # THE SINGLE-ROW CHECK COMES FIRST, before any chunking. The first version tested it only when
    # a row started a NEW chunk, so a lone oversized row -- the whole batch, one row, no boundary
    # to cross -- went straight to the pod and failed there on the shell instead of refusing here.
    # Caught by the pre-existing cap case, which is why that case stayed rather than being replaced
    # by the chunking one.
    lines = []
    for r in rows:
        line = json.dumps(r, ensure_ascii=False) + "\n"
        enc = len(base64.b64encode(line.encode("utf-8")))
        if enc > 90000:
            return 0, (f"a single row encodes to {enc} base64 chars, over the ~100KB argv cap -- "
                       f"it cannot be sent without splitting a JSON line")
        lines.append(line)

    chunks, cur = [], []
    for line in lines:
        if cur and len(base64.b64encode(("".join(cur) + line).encode("utf-8"))) > 90000:
            chunks.append(cur)
            cur = [line]
        else:
            cur.append(line)
    if cur:
        chunks.append(cur)

    sent = 0
    for i, lines in enumerate(chunks):
        b64 = base64.b64encode("".join(lines).encode("utf-8")).decode("ascii")
        cmd = f"cd {pod_root} && printf %s {b64} | base64 -d >> {rel}"
        r = subprocess.run(
            [os.path.expanduser("~/bin/pod"), cmd], capture_output=True, text=True,
            stdin=subprocess.DEVNULL
        )
        if r.returncode:
            return sent, (f"pod append failed on chunk {i + 1} of {len(chunks)} after {sent} "
                          f"row(s): {(r.stderr or 'rc=%d' % r.returncode).strip()[:120]}")
        sent += len(lines)
    return sent, None


def verify_pod_append(rel, rows, pod_root=POD_ROOT, reader=read_pod):
    """Read the pod back and confirm every row we sent is there, and that the pod's fold matches.

    A write whose success is inferred from rc=0 is not a verified write: the shell pipeline
    above can exit 0 with a truncated payload if base64 -d hits a short read, and a ledger
    that is silently short reads exactly like one that is correct. So this re-reads.

    IT USED TO DEMAND EACH SENT ROW BE THE POD'S LAST FOR ITS KEY, and that is wrong whenever a
    send carries more than one EVENT under one key -- which is the normal case, not an edge one.
    `missing` is the key-level set PLUS events_pod_lacks (:492), because a close is a second event
    under an existing key and would otherwise be unreachable. So a key can legitimately receive
    three events, of which only the last can possibly be last. MEASURED on C4's real push: 52 rows
    landed correctly, 348 -> 400 with 0 local-only remaining, and the old check reported 6 keys
    "not the one sent" (e1-21, e1-25, e1-26, e1-27, e1-29, e1-30) -- e1-26 received an `open`, a
    `done` at 05:57 and a `done` at 06:25, all three of which the pod lacked and all three of which
    it now has in that order. A verified write reported as failed is the same defect class as an
    unverified one reported as fine: the message stops meaning anything.

    What is actually required, and is checked here:
      1. every sent row is PRESENT on the pod -- nothing was truncated away, and
      2. the pod's fold for each touched key EQUALS the local fold -- the ordering landed such that
         the current state agrees, which is the property the transport exists to produce.
    (2) is the one that would catch a scrambled append that (1) alone would pass."""
    keys = _keys()
    text, err = reader(rel, pod_root)
    if err:
        return [f"cannot verify: {err}"]
    pod_rows, _ = parse(text)
    kf = keys[rel]
    bad = []
    present = [json.dumps(r, sort_keys=True, ensure_ascii=False) for r in pod_rows]
    from collections import Counter

    have = Counter(present)
    for r in rows:
        if not have.get(json.dumps(r, sort_keys=True, ensure_ascii=False)):
            bad.append(f"{kf(r)}: a sent row is absent after the append")
    pod_last, local_last = {}, {}
    for r in pod_rows:
        pod_last[kf(r)] = r
    lpath = os.path.join(ROOT, rel)
    if os.path.exists(lpath):
        lrows, _ = parse(open(lpath, encoding="utf-8").read())
        for r in lrows:
            local_last[kf(r)] = r
    for k in {kf(r) for r in rows}:
        if k in local_last and pod_last.get(k) != local_last[k]:
            bad.append(f"{k}: the pod's current row does not match this repository's after the send")
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
    for rel in _ledgers():
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
            # AND THE EVENTS UNDER SHARED KEYS, which `missing` cannot see: a close is a
            # second event under an existing key, so it is never `missing` and never applied
            # as a collision. Without this the push direction could not send the very rows
            # de-41 was written for (6e measured 3 appended and no_ghost_running still FAIL).
            missing = missing + events_pod_lacks(pod_rows, local_rows, keys[rel])
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
            # PRINT THE FIELD THAT ACTUALLY DIFFERS, not `status`/`result` by name: on
            # tasks.jsonl those two are absent from every row, so all 12 of these printed as
            # "pod [None] None / local [None] None" -- a report naming a row and then showing
            # nothing about it, which is worse than not printing it.
            _which = next((f for f in ("result", "reading", "evidence")
                           if not _empty(prow.get(f)) and _empty(lrow.get(f))), "result")
            _st = "status" if prow.get("status") is not None else "state"
            print(f"    CONTENT ONLY ON POD {k}  ({_which})")
            print(f"      pod   [{prow.get(_st)}] {str(prow.get(_which))[:88]}")
            print(f"      local [{lrow.get(_st)}] {str(lrow.get(_which))[:88]}")
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
    for rel in _ledgers():
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

    # THE SAME TWO CLASSES IN tasks.jsonl's FIELD NAMES (C4/DL-11, 2026-09-04). Widening the
    # transport to tasks.jsonl reported 53 contradictions, and measurement said 39 differ on
    # `state` alone (local closed, pod open -- staleness) and 14 have a `reading` on the pod and
    # none locally (content the repository lacks). Both were `contradicts` because this function
    # knew only `status`/`running` and only the `result` field, so it filed 53 rows as "a human
    # decides which is right" when 39 needed nothing and 14 needed copying.
    t_pod_open = {"id": "de-2", "state": "open", "task": "cap the token cache"}
    t_local_closed = {"id": "de-2", "state": "dropped", "task": "cap the token cache",
                      "drop_reason": "parked: experiment"}
    assert classify(t_local_closed, t_pod_open) == "stale", classify(t_local_closed, t_pod_open)
    t_pod_reading = {"id": "b0-20", "state": "open",
                     "reading": "launcher refuses a base with a different recorded hash"}
    t_local_bare = {"id": "b0-20", "state": "open"}
    assert classify(t_local_bare, t_pod_reading) == "result_only_on_pod", \
        classify(t_local_bare, t_pod_reading)
    # THE NEGATIVE CONTROLS, or the two clauses above would swallow real disagreements.
    # A pod row that is CLOSED differently from the local one is not staleness: the open-field
    # clause requires the POD side to be the open one.
    t_pod_done = {"id": "de-3", "state": "done", "task": "x"}
    t_local_dropped = {"id": "de-3", "state": "dropped", "task": "x"}
    assert classify(t_local_dropped, t_pod_done) == "contradicts", \
        classify(t_local_dropped, t_pod_done)
    # And two non-empty readings that differ stay a contradiction -- the content clause fires
    # only when the local side is EMPTY, never when it holds something else.
    t_both = {"id": "b0-21", "state": "open", "reading": "the gate reads the grant"}
    t_other = {"id": "b0-21", "state": "open", "reading": "the gate reads the config"}
    assert classify(t_both, t_other) == "contradicts", classify(t_both, t_other)
    # A pod row carrying BOTH a new reading and a contested result is a contradiction, not a
    # copy: `_contested` is what stops one absent field from licensing the whole row.
    t_mixed_local = {"id": "b0-22", "state": "open", "result": "40.0%"}
    t_mixed_pod = {"id": "b0-22", "state": "open", "result": "2.2%", "reading": "new"}
    assert classify(t_mixed_local, t_mixed_pod) == "contradicts", \
        classify(t_mixed_local, t_mixed_pod)

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
    assert len(rows) == len(_ledgers())
    assert all(r[5] for r in rows), f"a missing pod file reported no error: {rows}"
    assert all(not r[3] for r in rows), "a missing file must offer no rows to append"

    # A CLOSE MUST CROSS TO THE POD, and this is the case the row-level push could never
    # reach: a close is a second EVENT under an existing key, so diff_rows never calls it
    # `missing` and classify() calls it `stale`. Both correct, and between them the three
    # closed-but-running rows de-41 names were unreachable (6e measured 3 rows appended and
    # no_ghost_running still FAIL on the pod).
    ek = keys["runs/experiments.jsonl"]
    start = {"name": "r", "started": "t", "status": "running", "result": "", "ended": ""}
    close = {"name": "r", "started": "t", "status": "ok", "result": "3.6%",
             "ended": "2026-09-04 01:00"}
    got = events_pod_lacks([start], [start, close], ek)
    assert [r["status"] for r in got] == ["ok"], (
        f"a local close did not cross to a pod holding only the start: {got}. That is the "
        f"whole defect -- the pod row stays `running` and no_ghost_running stays FAIL there"
    )
    # ...and it folds CLOSED once appended, which is the property that matters rather than
    # the append itself.
    import sys as _sys
    _sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from exp import fold as _fold
    folded = {(r["name"], r["started"]): r for r in _fold([start] + got)}
    assert folded[("r", "t")]["status"] == "ok", f"appending the close did not fold closed: {folded}"

    # AND AN EVENT OLDER THAN THE POD'S LAST ONE MUST NOT CROSS. The pod already holds the
    # close; sending the start after it puts a `running` row LAST in the pod's file, and any
    # reader folding on position reads the finished run as open again. 6e left two such
    # events out by hand, which is why this direction is asserted and not assumed.
    got_back = events_pod_lacks([start, close], [start, close], ek)
    assert got_back == [], f"an event the pod already has was offered again: {got_back}"
    got_older = events_pod_lacks([close], [start, close], ek)
    assert got_older == [], (
        f"a start event older than the pod's close was offered: {got_older}. Appending it "
        f"would fold the row back to running by file position"
    )
    # A key the pod lacks ENTIRELY stays diff_rows' job, not this function's -- otherwise the
    # same row is offered twice and the pod gets a duplicate event.
    other = {"name": "q", "started": "u", "status": "ok", "result": "1.0", "ended": "x"}
    assert events_pod_lacks([start], [start, other], ek) == [], (
        "a whole key the pod lacks was returned by the event union as well as by diff_rows"
    )
    # TWO CLOSES WITH DIFFERENT VALUES stay a human's decision: it is a disagreement about a
    # measurement, not a missing event. classify keeps it in `contradicts`; the event union
    # must not smuggle it across as an append.
    close_b = {"name": "r", "started": "t", "status": "ok", "result": "9.9%",
               "ended": "2026-09-04 02:00"}
    smuggled = events_pod_lacks([start, close], [start, close_b], ek)
    assert smuggled == [], (
        f"a differing close was offered as an append: {smuggled}. Two closes under one key "
        f"are contradicts -- a human decides which measurement is right"
    )

    # SURVEY MUST ACTUALLY USE IT, asserted through survey(push=True) and not just on the
    # function. Both mutations that removed the wiring -- dropping the events_pod_lacks call
    # from the push path, and letting it return whole-key rows diff_rows already offers --
    # left every assertion above GREEN, because they all test the function in isolation. A
    # helper that works and is not called is the shape this whole file exists against
    # (measured 2026-09-04 by mutating the module).
    _ev_pod = [dict(start, name="ghost", started="g1")]
    _ev_loc = [dict(start, name="ghost", started="g1"),
               dict(close, name="ghost", started="g1"),
               dict(close, name="fresh", started="g2")]

    def _fake_pair(rel, pod_root):
        if rel != "runs/experiments.jsonl":
            return "", None
        return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in _ev_pod), None

    _tmp = __import__("tempfile").mkdtemp(prefix="ledger_union_")
    os.makedirs(os.path.join(_tmp, "runs"))
    for _rel in _ledgers():
        with open(os.path.join(_tmp, _rel), "w", encoding="utf-8") as _f:
            if _rel == "runs/experiments.jsonl":
                for _r in _ev_loc:
                    _f.write(json.dumps(_r, ensure_ascii=False) + "\n")
    _rows = {r[0]: r for r in survey(root=_tmp, reader=_fake_pair, push=True)}
    _off = _rows["runs/experiments.jsonl"][3]
    _sigs = sorted((r.get("name"), r.get("status")) for r in _off)
    assert _sigs == [("fresh", "ok"), ("ghost", "ok")], (
        f"survey(push=True) offered {_sigs}. It must offer BOTH the close under a shared key "
        f"(the event union) and the whole new key (diff_rows), each exactly once -- if the "
        f"close is missing the wiring is gone; if a name appears twice the two sources overlap "
        f"and the pod gets a duplicate event"
    )

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
    for rel in _ledgers():
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
    for rel in _ledgers():
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
    # failure this repo keeps buying. A SINGLE row over the cap still refuses -- it cannot be
    # split without splitting a JSON line, and half a line in an append-only ledger is a parse
    # error for every reader after it.
    big = [{"name": "x" * 70000, "started": "t"}]
    n, err = append_pod_rows("runs/experiments.jsonl", big, pod_root="/nonexistent")
    assert n == 0 and err and "argv cap" in err, (n, err)

    # BUT A BATCH over the cap is CHUNKED, not refused. Measured on C4's first real push: 52
    # tasks.jsonl rows carrying reading/why/evidence prose encode to 120,108 base64 chars, and the
    # refusal meant the transport could not move the one ledger that most needed it. Asserted
    # through a fake pod command that records each call, so the chunk BOUNDARIES are checked rather
    # than the total: every chunk must decode to whole lines, and their concatenation must equal
    # the payload.
    import base64 as _b64

    _calls = []

    def _fake_pod(argv, **kw):
        _calls.append(argv[-1])

        class _R:
            returncode = 0
            stderr = ""
        return _R()

    _rows = [{"id": f"t-{i}", "reading": "p" * 3000} for i in range(60)]
    _real_run = subprocess.run
    try:
        subprocess.run = _fake_pod
        n, err = append_pod_rows("runs/tasks.jsonl", _rows, pod_root="/work/aupai")
    finally:
        subprocess.run = _real_run
    assert err is None and n == 60, f"an over-cap BATCH must chunk, not refuse: {n}, {err}"
    assert len(_calls) > 1, f"60 rows of 3000 chars fit in one chunk; the case tests nothing: {len(_calls)}"
    _sent = ""
    for _c in _calls:
        _payload = _c.split("printf %s ")[1].split(" |")[0]
        _txt = _b64.b64decode(_payload).decode("utf-8")
        assert _txt.endswith("\n"), "a chunk does not end at a row boundary -- a split JSON line"
        assert all(json.loads(x) for x in _txt.splitlines() if x.strip()), "a chunk holds a partial row"
        _sent += _txt
    assert _sent == "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in _rows), (
        "the chunks do not reassemble into the payload")

    # THE VERIFIER HAD NO CASES AT ALL, which is how it shipped demanding that every sent row be
    # the pod's LAST for its key -- true only when a send carries one event per key, and `missing`
    # is the key-level set PLUS events_pod_lacks, so a key legitimately receives several. On C4's
    # real push 52 rows landed correctly (348 -> 400, 0 local-only left) and it reported 6 keys as
    # "not the one sent". A verified write reported as failed is the same defect as an unverified
    # one reported as fine.
    _vrel = "runs/experiments.jsonl"
    _vk = {"name": "vv", "started": "t1"}
    _open = dict(_vk, status="running")
    _mid = dict(_vk, status="ok", result="first")
    _final = dict(_vk, status="ok", result="second")

    def _pod_has(rel, pod_root, _rows=(_open, _mid, _final)):
        return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in _rows), None

    # THREE EVENTS UNDER ONE KEY, all sent, all present: this must pass. Under the old contract
    # the first two failed as "not the one sent".
    _saved_root = ROOT
    try:
        globals()["ROOT"] = os.path.join(d, "vworld")
        os.makedirs(globals()["ROOT"], exist_ok=True)
        os.makedirs(os.path.join(globals()["ROOT"], "runs"), exist_ok=True)
        with open(os.path.join(globals()["ROOT"], _vrel), "w", encoding="utf-8") as fh:
            for r in (_open, _mid, _final):
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        bad = verify_pod_append(_vrel, [_open, _mid, _final], reader=_pod_has)
        assert bad == [], f"three events under one key, all present, must verify: {bad}"
        # (1) A ROW THAT NEVER LANDED is caught. The pod holds only the first two.
        def _pod_short(rel, pod_root):
            return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in (_open, _mid)), None

        bad = verify_pod_append(_vrel, [_open, _mid, _final], reader=_pod_short)
        assert any("absent after the append" in b for b in bad), (
            f"a truncated append must be caught: {bad}")
        # (2) EVERY ROW PRESENT BUT THE ORDER SCRAMBLED, so the pod's current state disagrees with
        # this repository's. Presence alone passes; the fold comparison is what catches it.
        def _pod_scrambled(rel, pod_root):
            return "".join(json.dumps(r, ensure_ascii=False) + "\n"
                           for r in (_open, _final, _mid)), None

        bad = verify_pod_append(_vrel, [_open, _mid, _final], reader=_pod_scrambled)
        assert any("does not match this repository" in b for b in bad), (
            f"all rows present but folding to a different current state must be caught: {bad}")
    finally:
        globals()["ROOT"] = _saved_root

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
        f"pod_pull_ledgers selftest OK: {len(_ledgers())} ledgers keyed from ledger_audit.KEYS; "
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
        "truncating it. EVENT UNION (6e, 2026-09-04): a local close crosses to a pod holding "
        "only the start and folds closed there, while an event the pod already has, a start "
        "OLDER than the pod's close, a whole key that is diff_rows' job, and a second close "
        "with a different value are all refused -- the last staying a human's decision"
    )
    return 0


if __name__ == "__main__":
    sys.exit(_selftest() if "--selftest" in sys.argv else main())
