---
question: Does the launch/kill/monitor/milestone runner guarantee what its callers rely on?
status: recorded
source: b0-3 (fb tasking 2026-09-01); scripts/harness.py at 2df2689; 17 findings verified per-finding, 1 refuted; failing cases run on throwaway repos and read-only pod inspection, never the live run
---

# Runner review: launch, kill, monitor, milestone

Every finding below has a mechanical failing case that is red today. The class hunted is
the one this repo keeps producing: **a guard that tests a proxy for the property its
consumer needs.** Four of tonight's five recorded incidents are that shape, and so are
eleven of these seventeen.

Counts: **4 blockers, 9 silent-wrong, 4 wastes-hours.** By direction: **11 fail open**,
5 wrong-result, 1 ordering. Nineteen surfaces were checked and found sound.

Fail-open dominating is the finding behind the findings. A guard that refuses too much
gets fixed the first time it annoys someone; a guard that passes when it should refuse is
discovered only by an incident.

## Blockers

| # | surface | property needed | property tested | failing case |
|---|---|---|---|---|
| B1 | `cmd_launch` :7235 | no job starts on a card another job holds | `if not args.training ...` — occupancy read only for NON-training jobs, and only `cards.split(",")[0]` | stub every card occupied; a `--training` launch prints `launched ... on cards 0-6` and returns 0 |
| B2 | startup-gate kill :7318 | a `killed` row means the cards are free | that `killpg` did not raise; no wait, no re-probe, no GPU check | gate fails on `bash run_ddp.sh`; row says killed, torchrun survives holding 7 cards |
| B3 | `cmd_kill` :7547 | after `kill` returns 0 the job is dead | whether the recorded cmdline still matches | rename or restart shifts the cmdline: `host_pids` empty, every kill loop iterates nothing, `left` empty, reported clean |
| B4 | `readout_30b` main | a refusal reaches the caller as a nonzero exit | nothing — the return value was discarded | budget-mismatch refusal prints REFUSING and exits 0; `run_one` records a milestone with no verdict |

**B1 is the sharpest.** The occupancy check is skipped for exactly the launches that take
all seven cards, and `run point` already has the correct guard 600 lines away at :6617
(`_busy_cards(cards, settle=90)`). The protection exists in the file; the newer unified
launcher does not call it. Two concurrent pretrains are caught only incidentally, by
`run_ddp.sh`'s fixed `--master_port=29500` failing rendezvous — that costs a full gate
timeout and evaporates if anyone exports `PORT`.

**B4 was mine and is fixed** at e667c36. Two independent breaks: the CLI parsed
`--actual-tokens`/`--paired-tokens` and never passed them, so the refusal could not fire
at all; and `main()` discarded `readout()`'s False. Either alone hides the other. selftest
7 now asserts both halves against the real CLI.

## The name-vs-property defect, four instances

This is one bug written four times, and I wrote three of them today before finding the
fourth in the runner.

| where | asks | needs |
|---|---|---|
| `check_no_ghost_running` :577 | does any cmdline mention this run name | is the trainer alive |
| `_run_alive` :7698 | same, via `pgrep -f 'name {run}'` | same |
| `cmd_kill` monitor resolution :7558 | does any cmdline contain this integer | is this the monitor |
| my 8B file-watch and ledger grep | does this filename exist | does a scorable checkpoint exist |

**`_run_alive` is live and self-matching.** Its own probe command contains the pattern, so
`pgrep` matches the shell running it and the function returns True unconditionally.
Verified on the pod for a run that ended hours ago:

```
$ pgrep -af "name pretrain_15b_s1"
1159900 bash -lc pgrep -af "name pretrain_15b_s1" | head -3     <- only the probe
$ pgrep -af '[n]ame pretrain_15b_s1'
(nothing)                                                        <- correct
```

That is why the milestone watcher never exits, which is in turn why
`check_no_ghost_running` stays green over a dead run — **two fail-open guards holding each
other up.** The bracket form fixes the self-match; the deeper fix is to resolve liveness
from the pid file rather than any pattern.

## Ledger truthfulness

`_arm_monitor` :7083 closes **every** dead process as `--status ok`. It holds a pid, not
an exit code — it is detached and cannot `waitpid` — so OOM, NCCL timeout, a bad flag and
a clean finish are indistinguishable to it, and all four land in `runs/experiments.jsonl`
as success. Reproduced on a throwaway with a child exiting 1: the ledger recorded
`--status ok`. `check_score_matrix` then trusts `status == "ok"` and compounds it.

The monitor must not synthesise a verdict it cannot measure. `fail` with
`result: process gone, exit code unknown` is honest; `ok` is not.

Adjacent: an exec failure at :7296 leaves an open `running` row nobody closes, and
`settled()` folds rows by name rather than `(name, started)`, so a rerun inherits the
previous run's settled state.

## Gate derivation inverts

`_derive_gate_timeout` returns None when **no** cache is found, and the caller falls back
to 120s — while the work being timed has become a full retokenize. The emptier the cache,
the shorter the deadline. Worse, a **partial** cache passes silently: the derivation sizes
the gate from the caches that exist and ignores the domains that must be built. Refuse on
any missing domain, not only on all of them.

## G0–G4 status, my own guards against what shipped

| guard | status | evidence |
|---|---|---|
| G0 READY consumes the contract | **shipped** | launch_30b.sh:91-104 |
| G1a cache keyed on corpus fingerprint | shipped (pre-existing) | `.srcfp` in the conjunction |
| G1b cache keyed on the shuffle seed | **shipped, better than proposed** | `same_seed` via `_sample_seed()`; a separable `sample_seed` lets a seed sweep share one cache, which my version would have broken |
| G1c cache keyed on tokenizer behaviour | **not shipped** | no `.tokfp`; `vocab_fingerprint` still hashes `get_vocab()` only. Reproduced on the live tokenizer: moving one merge to the front leaves the fingerprint at `52300f002fcf0d06` and changes the token stream |
| G2a `launch_30b.sh` in training drift scope | **not shipped** | unreachable from the import BFS, so it classifies as docs |
| G2b tokenizer covered by the manifest | shipped | 1 manifest entry |
| G3 env compared to a verified baseline pre-launch | **partial** | `_env_fp_now` vs the checkpoint on auto-resume only; nothing at first launch |
| G4 resolved flags are the effective values | **shipped with a correction** | the `_switches` carve-out is right and my proposal was wrong: a `store_true` reads False both when absent and when defaulted, so blanket `is not None` would have written False over `Cfg.attn_res`'s True default and silently disabled AttnRes |

Two of my five guards were improved in implementation. Recording that is the point of the
column — a review that only counts what was adopted cannot see where the adopter was right
and the proposer wrong.

## One thing this review cannot tell you

Every finding here is a defect in a guard. **None of them is a defect in the training.**
The runner's job is to make the ledger true and the cards free, and the failures are all
of that kind: a row that says the wrong thing, a card held after a kill, a deadline that
inverts. Nothing here suggests the 30B run is training on the wrong data or with the wrong
flags — those properties are checked elsewhere and were verified separately tonight.

A worktree's hook is only as current as its last merge from main. `harness check` was
killed by an unhandled SIGALRM in my worktree for an hour after main had already fixed it
(2df2689), and the hook's advice — *"if this passes when rerun by hand, a concurrent write
raced a ledger read"* — sent me hunting a race that did not exist. Merge before diagnosing.
