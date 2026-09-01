---
question: "harness cross-review (e1-4): the pod and integration layer against P1-P8"
status: recorded
source: "e1-4 2026-09-01; failing cases in throwaway repos /tmp/mexempt and a --no-local clone of main; reviewer e1, code by de"
---

# Harness review: the pod and integration layer

Scope as assigned: `scripts/pod_push.sh`, `scripts/pod_drift.py`
(`write_manifest_index`, `--check-head`, `--check`), `scripts/pod_sync_check.sh`,
`scripts/hooks/pre-commit` including the tree check and merge exemption, and
`harness sync` (`_merge_jsonl` / `_verify_merge`). All of it de's code.

Every finding below was reproduced. Nothing here is read off the source alone.

## Findings

| id | file:line | principle | failing case | severity | proposed fix |
|---|---|---|---|---|---|
| E1 | `harness.py:4826` (`run_checks`) | P3 | `signal.alarm()` is armed with **no `SIGALRM` handler installed anywhere** in harness.py, so the default action kills the process. `except TimeoutError` can never fire. Reproduced: a child running the same shape exits `-14`, stdout and stderr both empty. This is fb's "exit -14 and no check named". | **principle** | **FIXED**, by de in `2df2689`, not by me. It refused the commit carrying this review, so I wrote a fix to deliver the audit; de had independently written a better one — scoped to `run_checks` and restoring the previous handler, where mine installed a global at import. Mine is dropped. What survives from me is `_selftest_check_timeout_skips`, which asserts an overrunning check SKIPs and names its deadline. `no_foreground_pod_training` was the slow check, measured 6.1 s against a 5 s default; de raised it to 15 s as a stopgap. |
| E2 | `hooks/pre-commit:88` | P3 | The merge exemption is `not merging`, where `merging` is only "`MERGE_HEAD` exists". It does not constrain **what is staged**. Reproduced in `/tmp/mexempt`: `git merge --no-commit other`, then `git add smuggled.txt`, then commit — accepted. `git diff --name-only HEAD^1 HEAD` gives `g.txt smuggled.txt` while the merge itself brought only `g.txt`. | **principle** | during a merge, refuse if the staged set differs from what the merge produced: compare `git diff --cached --name-only HEAD` against `git diff --name-only HEAD MERGE_HEAD` and refuse the extras |
| E3 | `hooks/pre-commit:83` | P3 | `AUPAI_INTEGRATION_TREE` is read from the environment with no validation, so setting it to any other path disables the tree check completely. Reproduced: `AUPAI_INTEGRATION_TREE=/somewhere/else git commit` landed a plain non-merge commit on `main` in the integration tree. It is documented nowhere — one occurrence in the whole repo, the line that reads it. | **principle** | it exists so the selftest's temp repos are not integration trees. Invert it: default to the real path, and only honour an override that points at an **existing** directory the process is actually inside; or drop it and key on `.git` layout instead |
| E4 | `pod_drift.py:507` | P3 | On a dev checkout `--check` prints "nothing to check" and **exits 0**. Any future caller that treats `--check` as a gate passes unconditionally off-pod. Mitigated today: both call sites (`pod_push.sh:96,127`) run it through `~/bin/pod` where `is_pod()` is true. | defect | `sys.exit(0)` there is the bug even though nothing exploits it yet — return a distinct code (e.g. 3) or require `--allow-noop`, so a gate cannot silently succeed |
| E5 | `pod_sync_check.sh:8,11,13` | P2 | `for f in $FILES` word-splits and `awk '$2==f'` splits on whitespace, so any scoped path containing a space is reported `MISSING` and skipped. Verified not exploitable **today**: `--list-scoped` returns 0 paths with a space. | nit | read paths NUL-separated, or `mapfile -t`; quote `"$f"` |
| E6 | `pod_drift.py:334` (`is_pod`) | P2 | `is_pod` is absence-of-`.git`, so a tarball extract, an rsync target, or any stray copy of the tree reads as **the** pod. Reproduced: `is_pod('/tmp/notapod')` is `True` for a directory holding two files. | nit | it is the right property for the real question and the blast radius is limited to where the script is invoked; if tightened, add a positive marker file the pod carries |

## What was tested and holds

The task named three history shapes for manifest generation. All three were run
against a `--no-local` clone of `main`, regenerating the manifest the way the
hook does — from the **index**, not the working tree.

| shape | result |
|---|---|
| fast-forward | `--check-head` OK, 200 files |
| merge, two branches touching different scoped files | OK after each of two successive merges |
| merge, both sides touching the same scoped file (manifest itself conflicts) | OK after resolve + regenerate |
| merge that changes a file with **nothing** in the index — the documented incident | OK; the `changed |= {p for p in head if sha_head != head[p]}` line at `pod_drift.py:185` is what fixes it, and it does |
| cherry-pick | OK, 200 files |

One false alarm I generated and then had to explain: regenerating the manifest
**before** staging the edit produces a stale row, because `write_manifest_index`
reads the index. That is correct behaviour and my test was wrong. Worth stating
because the same mistake produces a "manifest is broken" report that is really
an operator-order error.

Other properties confirmed by running them:

- `--check-head` fails **closed** on both an empty and a missing manifest (rc=1 each).
- `pod_push.sh` runs under `set -euo pipefail`, and its trailing pod-side `--check`
  is the last command, so a drift result fails the script. Verified with a stub.
- `push_one` refuses a file with uncommitted changes, a file absent from `main`,
  and a file whose content differs from `main` — three separate gates before any
  byte moves.
- `harness sync`'s `_merge_jsonl` unions and explicitly leaves folding to readers,
  which is P4-correct and consistent with the single `_exp_events` fold.
- Hook cost is instrumented per stage and totalled (`hook: total`), which is the
  P8 discipline; observed totals on real commits tonight were ~5 s.

## The two manifest-staleness incidents

Both are recorded in `write_manifest_index`'s docstring and both fixes are
present and effective:

1. **Rehash cost** (`4a7dd56`) — hashing all 174 paths took the hook past two
   minutes and `--no-verify` became habit. Fixed by reusing HEAD's sha for
   unchanged paths. The class is recomputed regardless, which is right: an import
   change elsewhere can reclassify an unchanged file.
2. **Merge staleness** (fb, 2026-09-01) — a merge changes files with nothing in
   the index, so the HEAD-reuse cache kept a stale row and `--check-head` refused
   while `--write-index` reported no diff. Fixed by adding paths whose HEAD sha
   already disagrees with the cached row. I reproduced the scenario and it now
   passes.

## Severity note

E1 was fixed twice in parallel: de in `2df2689` and me in `909b807`, within
about twenty minutes, neither aware of the other. I wrote mine only because the
bug refused the commit carrying this review — the audit could not be delivered
without a fix. de's is better and mine is dropped: theirs is scoped to
`run_checks` and restores the previous handler, where mine installed a global at
import, which is a side effect on every importer of harness.py including its own
selftests. I did not consider that.

My selftest survived, but only after it failed on de's code and I had to rewrite
it. It had asserted a handler was installed **at import** — true of my fix alone —
so it rejected the better implementation. It now asserts the property instead: an
overrunning check SKIPs and names its deadline. Verified to fail without a working
handler (rc 142, the process killed). A test that encodes one implementation
rejects its replacement, which is the same class as the four vacuous tests in last
night's retro, arriving from the opposite direction.
A per-check timeout that kills the process instead of skipping the check turns
one slow check into a commit refusal with no diagnosis, and the refusal text
tells the reader to rerun by hand, where it will likely pass. That combination
trains `--no-verify`, which is the P8 failure the whole hook is built to avoid.

E2 and E3 are both "the guard has an unconstrained escape". Neither is exploited
in tonight's history — I checked E2 against every merge on main today and found
no smuggled file — but both are one keystroke from being used, and E3 is
invisible because it is undocumented.

## Ceilings of this review

- I reviewed de's code and did not review my own contributions to the same files
  (`_exp_events`, the eval writers' `attest` call sites), which are 3b's and
  tilerl's scopes respectively.
- The pod-side behaviours were exercised locally against clones, not on the pod:
  `is_pod` was tested by constructing a directory without `.git`, not by running
  on the pod itself.
- E5 and E6 are latent. I state them as nits rather than defects because I could
  not construct a case where they bite today, and a review that inflates severity
  is as useless as one that misses things.
