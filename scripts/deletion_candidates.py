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

So the predicates live in the script. A listing is regenerated; a rule that only existed in the
head of whoever ran it is not (4c's ruling, 2026-09-08: "put the two predicates in the script
itself so the next person inherits them").

ADDING A PREDICATE IS THE EXPECTED KIND OF CHANGE. If you find a way a checkpoint can matter
that PROTECTIONS does not cover, add it here rather than remembering it at the prompt.
"""
import json
import os
import re
import sys
import time

ROOT = os.environ.get("AUPAI_ROOT", "/work/aupai")
MIN_BYTES = 1_000_000_000

#: Ledgers whose mention of a checkpoint means a recorded result depends on it. A result that
#: cites a checkpoint nobody can load is a result nobody can re-derive.
CITING_LEDGERS = (
    "score_matrix.jsonl",
    "b0_domain_loss_valrise.jsonl",
    "b0_domain_loss_resume1.jsonl",
    "b0_domain_loss_cap.jsonl",
    "review.jsonl",
    "prereg.jsonl",
    "milestones.jsonl",
)

_CKPT_RE = re.compile(r"ckpt_[\w.\-]+\.pt[\w.]*")


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


def protections(root=ROOT):
    """{basename: [reason, ...]} for every checkpoint something in the tree depends on."""
    out = {}

    def mark(name, reason):
        if name:
            out.setdefault(os.path.basename(str(name)), []).append(reason)

    # (1) PINNED. An explicit "do not lose this" in the milestones ledger.
    for r in _rows(os.path.join(root, "runs", "milestones.jsonl")):
        for key in ("ckpt", "pinned_as"):
            if r.get(key):
                mark(r[key], "PINNED in milestones.jsonl")

    # (2) CITED BY A RESULTS ARTIFACT. Missing from the first listing. A scored checkpoint is
    #     the only way to re-derive or dispute the score.
    for name in CITING_LEDGERS:
        for cited in _basenames(_read(os.path.join(root, "runs", name))):
            mark(cited, f"CITED by runs/{name}")

    # (3) RESUME SOURCE OF A RECORDED RUN. Also missing. This is the one that nearly cost
    #     .step9000: a resume source carries no pin, no hardlink and no open row once the run
    #     it seeded has finished, so all three original predicates read it as free.
    #     CLOSED ROWS COUNT. The trajectory is still reconstructible only while its source
    #     exists, and "the run finished" is precisely when the file stops looking needed.
    for r in _rows(os.path.join(root, "runs", "experiments.jsonl")):
        m = re.search(r"--resume\s+(\S+)", r.get("cmd") or "")
        if m:
            mark(m.group(1), f"RESUME SOURCE of {r.get('name')}")

    # (4) NAMED BY AN OPEN ROW. A job that has not finished may still read it.
    fold = {}
    for r in _rows(os.path.join(root, "runs", "experiments.jsonl")):
        fold[(r.get("name"), r.get("started"))] = r
    for r in fold.values():
        if r.get("status") == "running":
            for named in _basenames(r.get("cmd")):
                mark(named, f"named by OPEN row {r.get('name')}")
    return out


def listing(root=ROOT, min_bytes=MIN_BYTES):
    prot = protections(root)
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
        if st.st_nlink > 1:
            why.append(f"nlink={st.st_nlink}")
        rows.append({
            "file": name,
            "gb": st.st_size / 1e9,
            "age_days": (time.time() - st.st_mtime) / 86400,
            "protected_by": why,
        })
    rows.sort(key=lambda r: -r["gb"])
    return rows


def main(argv):
    root = argv[1] if len(argv) > 1 else ROOT
    rows = listing(root)
    free = [r for r in rows if not r["protected_by"]]
    held = [r for r in rows if r["protected_by"]]
    print(f"deletion candidates under {root}  (nothing is deleted by this script)\n")
    print(f"PROTECTED: {len(held)} files, {sum(r['gb'] for r in held):.0f} GB")
    for r in held:
        print(f"  {r['gb']:7.1f} GB  {r['file'][:52]:52s}  {'; '.join(r['protected_by'])[:70]}")
    print(f"\nNO PROTECTION FOUND: {len(free)} files, {sum(r['gb'] for r in free):.0f} GB")
    print("  -- 'no protection found' is a claim about the predicates in this file, not about")
    print("     the file's value. Check with its owner before deleting anything.")
    for r in free:
        print(f"  {r['gb']:7.1f} GB  {r['age_days']:6.1f}d  {r['file'][:56]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
