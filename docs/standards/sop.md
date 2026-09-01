---
question: How does this team work — who owns what, how work starts and ends, how conflicts resolve, and what may be believed?
status: recorded
source: rulings issued 2026-09-01; every rule names the incident that produced it
---

# SOP

Read this before your first action of a session. Every rule below cost something
today; the incident is named so you can judge whether the rule still applies.

**The one principle everything else follows from: a claim must be derived from the
thing it claims about.** A claim nothing recomputes cannot become false. It fails
silently and reads as healthy. Ten defects on 2026-09-01 were this and nothing else.

## 1. Who owns what

`python3 scripts/board.py who` — authoritative, in the repo, addresses are sockets.

**Address by socket, never by name.** `lessons-44` resolves to two live sessions and
only one is on this team. Three non-team sessions were given work on 2026-09-01
because an idle status in `ListAgents` was read as ownership. **Idle does not derive
membership.**

The controller (fb) rules and allocates; it does not operate. It does not touch GPUs,
the pod, or launches — four of its six errors that day were operational, all in
territory a peer knew better.

## 2. How work starts and ends

**The board is the shared state.** `board.py topics | show <t> | feed | post`.

- Every board command first prints what others posted since you last ran one.
  Reading is a side effect of writing; nobody has to remember.
- `find` / `rule` / `done` without `--artifact` is refused. **A claim nobody can
  check is chatter, and the board is where chatter collects first.**
- Post `block` when you are stuck and `done` when it clears. Two sessions waited on
  blockers that no longer existed that day; both were freed by someone reading the
  board, not by the controller.
- The patroller (98) pushes every new `rule` to the sessions it changes. **A ruling
  that was not delivered is worse than one not made: the controller thinks it is in
  force.**

**One anchor at a time.** When the controller names the anchor, work outside it stops
and is recorded, not done. Instrument work has no natural end; without an acceptance
line it grows until the experiment starves.

## 3. Conflicts

**Files.** Announce before editing `train.py` / `sft*.py` / `AGENTS.md` / hooks.
Commit within 30 minutes. Never `git checkout` or `git restore` a file you did not
write. Never `git stash` in the shared tree — that is five sessions' uncommitted work
in one operation.

**Cards.** The controller allocates; `runs/card_assignment.json` is the record.
**Idle is not free** — a card's owner is the job still running or the one the
controller has queued. Two collisions that day came from reading `nvidia-smi` as
authority.

**Kills.** By exact PID, after verifying the cmdline, never `pkill -f`. Then read the
card: **SIGTERM was sent is not the process left**, and only the second frees memory —
three deadlocked ranks held 72 GiB each through a TERM. A zombie holds a pid slot and
runs nothing; `pgrep -f X | wc -l` counted 1570 of them as live.

**Namespaces.** `tn exec` is the host; `~/bin/pod` is the container. Same path, two
directories, no error — the host's `/work/aupai` is a 9-file shell. **A PID from one
namespace does not exist in the other**; a launcher chained on a host PID fired
immediately and contended with a running probe. Cross-namespace identity is the GPU
UUID and the cmdline, nothing else.

**Merges.** The integrator (tilerl) merges into main. **When a check blocks you,
`git merge main` before diagnosing** — `no_foreground_pod_training` was patched four
times in one day and a session spent an hour diagnosing the version before the first
fix. Two sessions each saw the other's gate item as red while both were done.

## 4. Evidence

**A measurement's product is a file, not a message.** The gate reads files; it is the
only reader that does not forget, misremember, or queue in an inbox. Three gate items
sat red that day with the work finished and the number in someone's outbox.

**An artifact says what it did, not what it intended.** A record claiming a comparison
that never ran is more dangerous than a message, **because it will be cited**.
`gate_corpora` skipped the comparison when a mix carried no fingerprint and printed GO
anyway: 12 of 13 mixes, including the launch mix, were certified by a comparison that
never happened.

**Read a blob by its id, not by `<sha>:path`.** `git show <sha>:path` and
`git cat-file -p <sha>:path` will silently return a diff instead of the file. The
same 780-line file read as 29, 23, 25 and 594 lines across four attempts — **every
number plausible**, which is the danger: a 0 or an error would have stopped the
reader at the first attempt. Two people were nearly reported as having dropped
work they had not touched. Resolve first, then read:

```
git rev-parse <sha>:path        # -> blob id
git cat-file -p <blob-id>       # -> the file
```

**Every number carries its configuration, and that includes its resolution and its
extraction rule.** Six numbers were misused that day and every caveat was known at the
time — none was printed beside the number. A throughput field quantised to 1K was
quoted to three significant figures; a metric was compared across two tools whose FLOP
denominators differ; two of three plausible extraction rules produced a significant
difference the correct rule does not.

**A baseline is the strongest constant strategy, not random.** 8.13% on math looked
like signal against a 2.52% shuffle control and is below the 9.78% a model that always
answers "2" would score. **Answer distributions are skewed by default**, so a random
baseline systematically understates the floor.

**Before extrapolating from two points, check whether a third is already on disk.**
Every instance that day had the disconfirming number already available and unread.

**Before a two-arm test, name what else changed with the variable.** An A/B whose arms
ran in different machine-load windows was voided; the controller made the same error
on an MFU comparison that differed in five things.

## 5. Guards

**A guard is only as true as its inputs.** A 4-epoch ceiling stayed silent at a true
15.79 epochs because it was fed a supply figure never measured for the domain it
names. The guard was not broken.

**A guard only guarantees the one thing it checks.** An A/B guard that refuses
identical arms knows nothing about time windows; "the guard passed" was read as "this
A/B is clean".

**One red light per reason.** A permanent red is no signal — and a red with several
independent causes is worse, because it cannot be read at all. `restartability` was
red for three unrelated reasons at once: a 5s timeout, a phantom `.py` another
session's hook had just written into the directory it scans, and one genuinely
unlisted script. Any single fix left it red, so each looked ineffective and none was
finished. Split the causes before judging the light; a check that can fail three ways
should say which.

**A universal over a self-built population is only as true as the construction.**
"27 selftest-carrying files, all gated" while the real population was 36. Narrow the
matcher and it prints "0 files, all gated" — absurd at 0, unremarkable at 27.

**A check verified only against the current state is verified against the easy half.**
Three patches passed against a healthy pod and still failed the one captured broken
fixture.

**Every check carries a `broken()` that mutates a real artifact**, never a hand-written
world — a hand-written world shares the check's own assumptions. And **one broken world
is not enough**: a gate can be correct on one input and blind on another.

**A rule that must be remembered at the point of use has already failed. Only a rule
that refuses at the point of use holds.** Every rule broken that day was of the first
kind; every one that held was of the second.

## 6. Parallel work

**A blocked task gets more hands, not more waiting.** When the anchor task stalls,
the controller puts idle sessions on it. On 2026-09-01 the corpus build sat an hour
while three sessions were idle and the controller was still asking one person for a
status. **Silence and failure look identical from outside**, so a task that has not
reported in 30 minutes is treated as blocked, not as progressing.

**Parallel hands need one merge point, and it is the integrator.** Everyone commits in
their own worktree; nobody merges into main but the integrator. That is the only way
parallel work does not turn into a divergence problem — and divergence is what
actually cost time that day, not the work itself.

**Before adding a second person to a task, the first person's work must be committed
and merged.** Two people on an unmerged branch is not parallelism, it is a future
conflict.

## 7. Launch

`python3 scripts/launch_gate.py` decides, not a person. It is nine conditions computed
from artifacts, and it prints under every GO: *this is not a proof that the run is safe
to start; it is a proof that these nine specific failure modes are not present.*

**Each gate declares where its evidence lives** — code and config on main, corpus and
caches on the pod, drift on both. **The pod is authoritative where they disagree,
because the run happens there.** A run that reads fewer than nine refuses to print GO.

**Nothing checks that nine is the right nine.** That is a person, once, per launch. The
gate's job is to make that reading short, not to replace it.

**The code that runs is the newest code.** Sync is enforced at launch, not remembered:
a three-day run on a stale checkout is the most expensive error available, and it
produces a number we would read as the new recipe's.
