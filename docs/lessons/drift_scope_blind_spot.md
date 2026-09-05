---
question: Why was a tracked file that the pod reads at runtime stale there while pod_sync_check reported "in sync"?
status: recorded
source: 4c's report 2026-09-05 during the credential rewrite; scripts/pod_drift.py SCOPE (:66) vs scripts/hooks/pre-commit ALLOWED (:52)
---

# Shape: a tracked file outside the drift scope was stale on the pod with the gate green

**Reported by** 4c, 2026-09-05, during the credential rewrite.
**Filed by** de, for 44.

## What happened

`data/probes/api_cloze.jsonl` — the file that carried a live third-party Postgres credential —
was present on the pod at the pre-rewrite content while `scripts/pod_sync_check.sh` reported
`pod in sync (463 files)`. The file had to be pushed by hand. The gate was not broken and it
was not lying: the file is not in `pod_drift.SCOPE`, so the manifest never described it, and a
manifest that does not list a path cannot assert anything about it.

## Why the scope did not cover it

`SCOPE` admits `data/` only through three named patterns:

    data/mix_*.json
    data/tokenizer.json
    data/ledger_schema.json

Everything else under `data/` is deliberately out, because `data/corpus/*` is 81 GB+ of bytes
that are gitignored and must never enter a manifest. So the exclusion is correct for the
directory and wrong for one new subdirectory inside it. `data/probes/` was created on 2026-09-05
by the same commit that leaked, and the ALLOWED list in `scripts/hooks/pre-commit` was extended
to admit it as a tracked path — but the two lists are maintained separately and nothing joins
them.

## The shape

**A path can be tracked, committed, read at runtime on the pod, and outside the drift manifest
at the same time. The drift gate then reports green about a file it has never looked at, and
"in sync (463 files)" is a count of the files in scope, not of the files that matter.**

This is the same class as `data/pod_head_manifest.txt`'s own rule — "the manifest asserts that
the files it lists match, never that unlisted ones are absent" — one level out: unlisted files
are not only allowed to be absent, they are allowed to be STALE, and the number in the success
line makes the coverage look total.

## Why no existing check catches it

- `pod_drift --check` compares what the manifest lists. Out of scope, out of the comparison.
- `pod_sync_check.sh` is a wrapper over the same manifest. Same blind spot, plus a reassuring count.
- The pre-commit `ALLOWED` list decides what may be COMMITTED under `data/`, not what must be
  SYNCED. A path can be in one and not the other, and today `data/probes/*.jsonl` was.
- `reachability.py` sees a doc or code citation, not a manifest membership.

## The joinable invariant

Every tracked path that a script on the pod READS at runtime must be in `SCOPE`. Both sides of
that are machine-readable: `git ls-files` gives tracked paths, and the pre-commit `ALLOWED`
patterns already enumerate the tracked non-code `data/` paths someone deliberately admitted. A
check that intersects `ALLOWED` with `SCOPE` and FAILs on a path in the first and not the second
would have caught this at the commit that created `data/probes/`, without needing to know that
`eval/api_cloze.py` reads it.

Stated as its weaker, certain form: **a tracked path admitted to `data/` by `ALLOWED` and absent
from `SCOPE` is a defect until someone writes down why the pod does not need it.**

## Immediate fix

`data/probes/*.jsonl` added to `SCOPE`. That closes this instance; the check above is what closes
the class, and it is not written yet.
