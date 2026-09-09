#!/usr/bin/env python3
"""Checkpoint deletion candidates, with the reason each protected file is protected.

MEASURES ONLY. Nothing here deletes, and nothing here is a recommendation: a candidate is a
file this script could find no protection for, which is a statement about the predicates below
and not about the file's value.

WHY THIS EXISTS AS A SCRIPT AND NOT AS A COMMAND SOMEONE RETYPES. On 2026-09-08 the pod hit
95% and the first listing I produced scored "protected" as pinned OR hardlinked OR named by an
open experiments row. Under those three predicates
`ckpt_1.5b-a0.2b-e48_8b.pt.step9000` came out UNPROTECTED -- 6.1 GB, no pin, nlink 1, no open
row -- and it is the resume source of the entire 30B trajectory and the baseline row in the
endpoint comparison filed an hour earlier. The listing would have offered for deletion the one
checkpoint the endpoint is compared against.

The file was never at risk of being wrong about; the LISTING was wrong about itself. Three
predicates answered a four-predicate question and the output gave no hint that a fourth
existed -- every row was well-formed, the omission had no representation. Adding the two
missing predicates moved 53 GB from candidate to protected.

AND THEN THE FIX REPEATED THE DEFECT ONE LEVEL UP (3b, 2026-09-08). The citation predicate
scanned a hand-written tuple of SEVEN ledgers while the pod holds 78 runs/*.jsonl, so the
predicate that exists to catch "a result depends on this file" was itself asking a narrower
question than the one it was added to answer -- and its output looked exactly as complete as
before. Four checkpoints came out NO PROTECTION FOUND while cited: two by tasks.jsonl, one by
friction.jsonl, and `ckpt_b0_mem_m2.pt.interrupt.step36` by experiments.jsonl, a file this
script ALREADY OPENS for the resume-source predicate and still did not scan for citations.
The glob below is the fix: the population is every ledger, chosen by the filesystem, not a
list that has to be remembered when someone adds a ledger.

The same shape a third time, in the resume predicate: it matched `--resume PATH` and not
`--resume=PATH`, so a run launched with the equals form protected nothing. Both forms now.

ADDING A PREDICATE IS THE EXPECTED KIND OF CHANGE. If you find a way a checkpoint can matter
that protections() does not cover, add it here rather than remembering it at the prompt. The
selftest below cuts one predicate at a time and requires the listing to CHANGE, so a predicate
that has stopped matching anything is a failure rather than a silent no-op.
"""
import glob
import json
import os
import re
import sys
import time

ROOT = os.environ.get("AUPAI_ROOT", "/work/aupai")
MIN_BYTES = 1_000_000_000

_CKPT_RE = re.compile(r"ckpt_[\w.\-]+\.pt[\w.]*")

#: The `ckpt_NAME.pt` prefix of a `ckpt_NAME.pt.step9000` / `.ep1` / `.interrupt.stepN` sibling.
_BASE_RE = re.compile(r"(ckpt_[\w.\-]+?\.pt)")

#: Both spellings argparse accepts. `--resume=PATH` was missed by the first version, which is
#: the same defect as the seven-ledger tuple: a pattern narrower than the thing it names.
_RESUME_RE = re.compile(r"--resume[\s=]+(\S+)")


def _basenames(text):
    return {os.path.basename(m) for m in _CKPT_RE.findall(text or "")}


def _read(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError:
        return ""


def _rows(path):
    for line in _read(path).splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except ValueError:
            continue


def ledgers(root=ROOT):
    """Every runs/*.jsonl, sorted. The population is the filesystem's, not a list I maintain."""
    return sorted(glob.glob(os.path.join(root, "runs", "*.jsonl")))


def protections(root=ROOT, skip=()):
    """{basename: [reason, ...]} for every checkpoint something in the tree depends on.

    `skip` names predicates to leave out, by the numbers below. Only the selftest passes it:
    cutting a predicate must change the listing, and a cut that changes nothing means the
    predicate is no longer carrying its case.
    """
    out = {}
    exp = os.path.join(root, "runs", "experiments.jsonl")

    def mark(name, reason):
        if name:
            out.setdefault(os.path.basename(str(name)), []).append(reason)

    # (1) PINNED. An explicit "do not lose this" in the milestones ledger.
    if 1 not in skip:
        for r in _rows(os.path.join(root, "runs", "milestones.jsonl")):
            for key in ("ckpt", "pinned_as"):
                if r.get(key):
                    mark(r[key], "PINNED in milestones.jsonl")

    # (2) CITED BY ANY LEDGER. A result that cites a checkpoint nobody can load is a result
    #     nobody can re-derive. Scanning ALL of runs/*.jsonl also subsumes the open-row
    #     predicate's grep of experiments.jsonl -- (4) is kept anyway because it says WHY,
    #     and a reason a human can act on is the output's whole product.
    if 2 not in skip:
        for path in ledgers(root):
            for cited in _basenames(_read(path)):
                mark(cited, f"CITED by runs/{os.path.basename(path)}")

    # (3) RESUME SOURCE OF A RECORDED RUN. A resume source carries no pin, no hardlink and no
    #     open row once the run it seeded has finished, so all three original predicates read
    #     it as free. CLOSED ROWS COUNT: the trajectory is reconstructible only while its
    #     source exists, and "the run finished" is precisely when the file stops looking
    #     needed.
    if 3 not in skip:
        for r in _rows(exp):
            m = _RESUME_RE.search(r.get("cmd") or "")
            if m:
                mark(m.group(1), f"RESUME SOURCE of {r.get('name')}")

    # (4) NAMED BY AN OPEN ROW. A job that has not finished may still write to or read it.
    if 4 not in skip:
        fold = {}
        for r in _rows(exp):
            fold[(r.get("name"), r.get("started"))] = r
        for r in fold.values():
            if r.get("status") == "running":
                for named in _basenames(r.get("cmd")):
                    mark(named, f"named by OPEN row {r.get('name')}")
    return out


def listing(root=ROOT, min_bytes=MIN_BYTES, skip=()):
    prot = protections(root, skip=skip)
    rows = []
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if ".pt" not in name or not os.path.isfile(path):
            continue
        st = os.stat(path)
        if st.st_size < min_bytes:
            continue
        why = list(prot.get(name, []))
        # (5) HARDLINKED. Some other name points at these bytes; unlinking this one frees
        #     nothing, and the other name is usually a pin.
        if 5 not in skip and st.st_nlink > 1:
            why.append(f"nlink={st.st_nlink}")
        rows.append({
            "file": name,
            "gb": st.st_size / 1e9,
            "age_days": (time.time() - st.st_mtime) / 86400,
            "protected_by": why,
            # NOT A PROTECTION, AND DELIBERATELY NOT ONE. `ckpt_X.pt.step500` when `ckpt_X.pt`
            # is cited: an intermediate step of a run whose endpoint someone scored is often
            # genuinely deletable, so promoting this to a predicate would protect 19 of the 44
            # free files here and make the listing useless. But it is the difference between
            # "nothing refers to this file" and "nothing refers to THIS STEP of a run that is
            # referred to", and only the second is safe to act on without asking. 3b named
            # ckpt_ab_fp32logits_base.pt.step500 as cited; measured, no ledger names it -- only
            # its base does. The row carries the fact so the reader can tell the two apart.
            "family_cited": None,
        })
    prot_names = set(prot)
    for r in rows:
        if r["protected_by"]:
            continue
        m = _BASE_RE.match(r["file"])
        base = m.group(1) if m else r["file"]
        if base != r["file"] and base in prot_names:
            r["family_cited"] = base
    rows.sort(key=lambda r: -r["gb"])
    return rows


def _selftest(root=ROOT):
    """Assert the glob's coverage against the REAL ledgers, not a fixture.

    Built from the real tree deliberately. The defect this script keeps reproducing is a
    population narrower than the question, and a fixture is a population I choose -- a
    synthetic ledger set would have passed every version of this file, including the one that
    scanned seven of 78 ledgers.

    THE FIRST VERSION OF THIS SELFTEST ASSERTED THE WRONG THING: that cutting any predicate
    must free some file. It fired on 1, 3 and 4, and the measurement showed the assertion was
    wrong rather than the code. Predicates 1 and 3 read milestones.jsonl and experiments.jsonl,
    which the glob ALSO reads, so their protection is necessarily a subset of the glob's -- 33
    and 13 files named, 0 of them uniquely. That subsumption is not redundancy to delete; it is
    the invariant worth asserting, and it is exactly what the seven-ledger tuple broke.
    """
    if not os.path.isdir(os.path.join(root, "runs")):
        print(f"deletion_candidates selftest SKIP: no {root}/runs (not the pod)")
        return 0

    found = ledgers(root)
    assert len(found) >= 7, f"only {len(found)} runs/*.jsonl found under {root}"

    base = listing(root)
    if not base:
        print(f"deletion_candidates selftest SKIP: no >={MIN_BYTES/1e9:.0f} GB *.pt under {root}")
        return 0
    base_free = {r["file"] for r in base if not r["protected_by"]}

    # (A) THE GLOB IS LOAD-BEARING. Cutting the citation predicate must free files; if it frees
    #     none, the glob is matching nothing and every other check here is vacuous.
    cut2 = {r["file"] for r in listing(root, skip=(2,)) if not r["protected_by"]}
    assert cut2 - base_free, (
        "cutting the citation predicate freed no file, so the glob is protecting nothing -- "
        f"it matched {len(found)} ledgers, so it is finding files but not citations in them.")

    # (B) THE GLOB SUBSUMES EVERY SINGLE-LEDGER PREDICATE. Predicates 1, 3 and 4 read ledgers
    #     that runs/*.jsonl contains, so any file they protect must also be cited-by. A file
    #     protected ONLY by one of them means the glob failed to read that ledger.
    #
    #     THIS IS THE ASSERTION THAT CATCHES THE ORIGINAL DEFECT, and it was verified by
    #     running the defect rather than by reasoning about it: restoring the seven-ledger
    #     tuple leaves 4 files protected only by predicate 3 -- ckpt_p200m_4b_0902.pt.
    #     interrupt.step832, ckpt_p47_s1.pt.step50, ckpt_t38_kill.pt.step40,
    #     ckpt_twin_nocursor.pt -- because experiments.jsonl was not among the seven. It fires
    #     on the resume predicate and not on 1 or 4: milestones.jsonl WAS in the tuple, and
    #     predicate 4 currently matches nothing at all. So this catches the real defect through
    #     one predicate only, which is worth knowing -- had experiments.jsonl happened to be in
    #     the hardcoded list, no assertion here would have fired while 71 ledgers went
    #     unscanned. Assertion (A) plus the printed ledger count are what cover that.
    by_glob = set(protections(root, skip=(1, 3, 4)))
    for p, label in ((1, "PINNED/milestones.jsonl"), (3, "RESUME/experiments.jsonl"),
                     (4, "OPEN ROW/experiments.jsonl")):
        solo = set(protections(root, skip=tuple(q for q in (1, 2, 3, 4) if q != p)))
        missed = sorted(solo - by_glob)
        assert not missed, (
            f"predicate {p} ({label}) protects {len(missed)} file(s) the citation glob does "
            f"not: {missed[:5]}. The glob reads every runs/*.jsonl, so it should already cite "
            f"anything named in that ledger -- this means it is not reading it.")

    # (C) BOTH RESUME SPELLINGS. Asserted on strings, not on the tree: the tree may hold only
    #     one of the two forms, and then the equals case is untested exactly when it matters.
    assert _RESUME_RE.search("train.py --resume /work/a.pt").group(1) == "/work/a.pt"
    assert _RESUME_RE.search("train.py --resume=/work/a.pt").group(1) == "/work/a.pt"

    # (D) THE FAMILY ANNOTATION IS NOT A PROTECTION. It must never move a file out of the free
    #     list -- if it does, someone has promoted it to a predicate and 19 of the 44 free
    #     files here silently became protected.
    fam = [r for r in listing(root) if r["family_cited"]]
    assert all(not r["protected_by"] for r in fam), (
        "a file carries family_cited AND a protection reason; family_cited is annotation only "
        "and must be set on free rows exclusively.")

    # A predicate matching nothing is REPORTED, not failed. It can be correct and currently
    # unmatched -- (4) names nothing whenever no run is open, which is most of the time.
    counts = {p: len(protections(root, skip=tuple(q for q in (1, 2, 3, 4) if q != p)))
              for p in (1, 2, 3, 4)}
    idle = [p for p, n in counts.items() if n == 0]
    print(f"deletion_candidates selftest OK: {len(found)} ledgers, {len(base)} checkpoints, "
          f"{len(base_free)} unprotected ({len(fam)} of them intermediate steps of a cited "
          f"run); predicate coverage {counts}"
          + (f"; matching nothing right now: {idle}" if idle else ""))
    return 0


def main(argv):
    if "--selftest" in argv:
        return _selftest(argv[1] if len(argv) > 1 and not argv[1].startswith("-") else ROOT)
    root = argv[1] if len(argv) > 1 else ROOT
    rows = listing(root)
    free = [r for r in rows if not r["protected_by"]]
    held = [r for r in rows if r["protected_by"]]
    print(f"deletion candidates under {root}  (nothing is deleted by this script)")
    print(f"  citation scan covers {len(ledgers(root))} ledgers under {root}/runs\n")
    print(f"PROTECTED: {len(held)} files, {sum(r['gb'] for r in held):.0f} GB")
    for r in held:
        print(f"  {r['gb']:7.1f} GB  {r['file'][:52]:52s}  {'; '.join(r['protected_by'])[:70]}")
    print(f"\nNO PROTECTION FOUND: {len(free)} files, {sum(r['gb'] for r in free):.0f} GB")
    print("  -- 'no protection found' is a claim about the predicates in this file, not about")
    print("     the file's value. Check with its owner before deleting anything.")
    print("  -- (family) marks an intermediate step whose BASE checkpoint is cited. Not a")
    print("     protection; it means a recorded result depends on that run, not on this step.")
    for r in free:
        tag = f"  (family: {r['family_cited']})" if r["family_cited"] else ""
        print(f"  {r['gb']:7.1f} GB  {r['age_days']:6.1f}d  {r['file'][:56]}{tag}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
