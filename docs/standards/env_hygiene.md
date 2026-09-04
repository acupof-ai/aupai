---
question: How does this environment stay clean without a person watching it, and what does that cost?
status: design
source: audit findings PR-10, PR-11, PR-1/PR-2, cleanup C1 (ff035f77); measurements 2026-09-04 03:33-07:40Z
---

# Environment hygiene

User order 2026-09-04: *"环境的问题我们应该想个办法，还是有一个容器什么的，把这个东西可以自动清理，包括 Host 上现在也不太干净。"*

Design only. Nothing here is built; `harness sweep` is db's, the container change is the
user's. Where a number appears it was measured, not estimated.

## What is actually dirty, in order of size

| symptom | measured | growth | can it be fixed from inside? |
|---|---|---|---|
| **zombie processes** | **3,958 of 4,306** (92%) | ~300/day, 2 in a 10-min window | **no** — see §1 |
| orphaned watchers | 307 `tail -F` + 5 stale `tail -f` | one per ~305 s for 24 h, then stopped | yes — swept, C1 |
| non-terminating wait-loops | 3, ages 4.58–4.60 d | — | yes — swept, C1 |
| host leftovers | **none found** | — | n/a — see §4 |

C1 removed 314 processes (`ff035f77`). The zombies remain and cannot be removed by anything
running inside the container.

## 1. The container: PID 1 must reap

**Why nothing inside can fix this.** A zombie is already dead; the kernel keeps only its
PID slot and exit status. That slot is released when its **parent** calls `wait()`. No
signal reaches a zombie — `kill -9` on one is a no-op — so the only actor who can clear it
is the parent. All 3,958 have `PPid 1`, and PID 1 in this container is `sleep infinity`: a
shell builtin that blocks in `sleep` and never calls `wait()`. There is no namespace from
which this is reachable.

**A zombie keeps its `/proc` entry, and that has already cost time.** `[ -d /proc/<pid> ]`
and `os.path.exists("/proc/<pid>")` read **true** for a process that has exited — db
measured a queue waiting 31 minutes on one. **Liveness is `stat` field 3 not in `Z`, never
the presence of `/proc/<pid>`.** Same family as the two other identity traps in this
document: `nvidia-smi` host pids resolved in the container namespace (§2e), and the
container's default cwd read as ownership (§5).

**The sweeper counts zombies and never lists them as candidates.** They are not a
process-hygiene problem a sweeper can fix; the count is a standing environment fact and
the fix is §1's init change.

**Severity is bounded, and the numbers say so.** `pids.current` 4,647 against a cgroup
`pids.max` of `max`; 4,307 PIDs against `pid_max` 4,194,303; zombie RSS 0. Nothing is near a
limit and no work is blocked. This is unbounded growth with a 4,000 head start, not an
outage — which is why it is filed S3 and why the fix is scheduled rather than urgent.

**The fix, three forms, cheapest first:**

1. **The spawner reaps its own children.** Free, no restart, but it only covers processes
   we launch — the 3,095 `python3` zombies suggest most are ours, so this would stop most
   of the bleeding without touching PID 1.
2. **A reaping PID 1**: `tini`, `dumb-init`, or `exec bash -c 'while :; do wait -n; done'`
   as the container entrypoint. Complete, and the only form that also handles processes we
   do not launch.
3. **A container restart** clears all 3,958 at once but does not prevent the next 3,958.

**The one-time cost, stated because it is the user's to accept.** Changing PID 1 requires
recreating the container. **That kills every process in it**, including the user's own
tileRL jobs (three checkouts, `/work/tl-ab`, `/work/tl-rl`, `/work/tilerl`, seen running on
cards 1, 2 and 7 today) and any aupai training run alive at that moment. It is therefore
**scheduled by the user, not by us**, and form 1 is available meanwhile at zero cost.

## 2. The sweeper — three classes in the container, two answered host-side

`harness sweep` (db builds it). It kills **only** by exact PID, **only** the classes below,
and writes every decision to `runs/sweeper.jsonl`. `--execute` defaults **off**.

**Split, per 6e's ruling.** The container sweeper implements **(a), (b)-file-existence and
(c)**. **(d)** and **(e)** are protections answered from the **host view via `tn exec`** —
(e) keyed on GPU UUID + cmdline — because inside the container neither cgroups nor
`nvidia-smi` pids discriminate (see (e)). Zombies are **counted and reported**, never
listed as candidates.

**The rule that outranks the classes: a class that cannot be positively established is
reported `unclassified` and never killed.** A sweeper that guesses is worse than the
orphans it removes.

### (a) Orphaned watcher — output reaches nobody

Either disjunct is sufficient:

- **a1** — `fd/1` is a pipe and **no other live process holds that inode**. A pipe with one
  endpoint is unreadable forever, so its writer is delivering to nobody.
- **a2** — `fd/1` is a regular file, the command is a `tail`/`watch` form, **and class (c)
  holds for its target.** a2 alone is not sufficient: a watcher writing to a file for a
  live run is doing its job.

Two implementation facts, both learned by getting them wrong on 307 processes today:
**exclude the scanning process itself** from the holder set (it holds its own fd; I got a
false "1 shared" twice, and the pid differing between runs is what exposed it), and
**re-check that the other holder is alive**, because a dead holder's fd survives the walk.

### (b) Wait-loop whose producer is gone

Matches **only** `[ -f X ]`, `[ -e X ]`, `[ -d X ]` with a single literal path. Compound
conditions are not parseable in general and are `unclassified`.

Test: **X does not exist, and X's parent directory's newest mtime predates the loop's own
start.** That is positive evidence the producer is gone — anything still producing X would
have moved the directory since the waiter started.

**Do not test `ppid == 1`.** A reparented orphan and a legitimate `setsid` job both have
ppid 1; the parent is not evidence. Measured on today's three: `data/code_supply/` last
written 2026-08-30T14:47Z, loops polling every 8–12 s for 4.6 days.

### (c) Tail on a finished run's log

- "Finished" is decided by **`exp.fold`** — the current row per `(name, started)` — **never
  the last ledger line**. A last-line reading goes vacuous the moment someone appends a
  close, and would call a live run closed.
- Mapping is **`runs/<name>.log` → the row whose `name` equals `<name>` exactly**. No fuzzy
  match; an unmatched basename is `unclassified`.
- Extra guard: the log's mtime must be **older than the row's end**. A log still growing
  after its row closed means the row is wrong, not that the tail is stale.

### (d) Ours and accounted → never swept

**Rests on the experiments row. A `card_claim` claim is corroboration, never the test.**

db measured why: **10 of the 11 GPU entry points that load a checkpoint never call
`card_claim`, and 9 cannot even name their card.** `runs/claims/` held one file while lane
jobs ran all day. A sweeper resting on claims would classify real work as sweepable. This
relaxes when de-55 lands.

### (e) Foreign or GPU-holding → NEVER, and not by prose

**Do not reuse `FOREIGN_MARKERS`.** `card_claim.foreign_cards` (`card_claim.py:316`)
decides "foreign" by matching hand-written strings — `"NEVER TAKE"`, `"another container"` —
out of `runs/card_assignment.json`. Audit finding PR-1: it called card 7 another container's
while `/proc/<pid>/cgroup` was **byte-identical to our own trainer's**. The job was in *our*
container from a *different repository*. **Repository, container and namespace are three
properties**; that marker fuses them and is wrong on any case that separates them.

The positive test, which needs no prose:

> **Resolve a pid only in the namespace that produced it.** `nvidia-smi` reports **host**
> pids — read them via `tn exec` against host `/proc/<hostpid>/cgroup`. Container `ps`
> pids resolve only inside. **Never intersect the two sets.**

So: skip any process whose host cgroup names a container id other than ours, and skip any
host pid in `nvidia-smi --query-compute-apps`.

**PRECONDITION, and it decides where the sweeper may run at all.** db measured, and I
confirmed: **there is exactly ONE distinct cgroup across every live process in the
container** — PID 1, our jobs, and both tileRL trainers are byte-identical at
`kubepods/besteffort/pod95a05e32…/827d3e58…`. Inside the container the cgroup has **zero
discriminating power**; it cannot separate ours from tileRL's because at that level they
are not separate. And container `nvidia-smi` reports host pids (3487226, 3547785 in db's
read) that resolve in no container `/proc`.

By this document's own "cannot resolve → `unclassified`" rule, **a container-side sweeper
classifies every process as unclassified and sweeps nothing.** Class (e) is therefore not a
rule about pids, it is a **constraint on the sweeper's location**:

> `harness sweep` must read host `/proc` through `tn exec` for classes (d) and (e). Run
> purely inside the container, it is correct but inert.

**Ruling (6e, 2026-09-04): (e) is not a container-sweeper class at all.** It is a
**host-view check via `tn exec`, keyed on GPU UUID + cmdline**, not on pids and not on
cgroups. The container sweeper therefore covers **(a), (b)-file-existence and (c) only**,
against the 314 processes recorded this morning; ownership of a GPU process is answered
host-side or not at all. `card_assignment.json` prose may remain an *additional* exclusion,
never the test.

Measured twice today, once by db and once by me: intersecting container pids with
`nvidia-smi` pids resolves **none** of our own live training ranks and marks everything
foreign.

### Kill discipline

**Re-read `/proc/<pid>/cmdline` immediately before each signal and abort that pid if it no
longer matches.** PIDs are reused and a scan is minutes old by the time it acts. C1 killed
314 and this guard skipped exactly one (pid 277143, no longer the expected tail) — the skip
is the feature.

### What `runs/sweeper.jsonl` records

Per run: **killed and skipped with the reason for each** (a kill count alone cannot be
audited), and **the selector beside every count**. PR-11 reported 153 rolling checkpoints
where b0's glob gave 149; the four-file gap was pure selector, not disagreement.

### Scope, and the paths that are never in it

**In scope: `/work/aupai` and the container's tmp/scratch classes only.**

**Never in scope, by path:**

- `/data01/aupai/backup` and `/data01/aupai/backup.log` — see §3.
- Anything on `/data00`, `/data01`, `/data02`, `/data03` until a class explicitly names it.
- Any GPU process, any foreign cgroup, per (e).

## 3. The backup, and the refresh decision that is the user's

`/data01/aupai/backup` is a **365.2 GiB rsync of the project tree**, 4,685 files, taken
2026-08-30 19:32–20:16Z. `/work` is a kubernetes emptyDir destroyed on pod restart;
`/data01` is a real NVMe mount. **This is the only durable copy of the project on this
machine.**

**What the old rsync excluded — derived from the trees, because the log does not say.**
`backup.log` contains only rsync progress lines plus `BACKUP_DONE` and `RESTRUCTURED`; the
invocation and any `--exclude` flags are not in it. Comparing directory by directory
against the live tree:

| directory | in backup | live now | read |
|---|---|---|---|
| `data/` | 289.8 GB | 579.8 GB | **half** — the corpus has since doubled |
| `runs/` | **5.6 MB** | 24.5 GB | **effectively excluded** — logs and traces are not in it |
| `bench_eff/` | 996 MB | 3.04 GB | partial |
| `retired_pre0830v2/` | 6.19 GB | 6.19 GB | **identical** |
| `scripts/`, `docs/`, `facts/`, `eval/`, `datagen/` | 0.15–1.0 MB each | 0.66–5.1 MB each | present, ~5× smaller |

No top-level entry is in one tree and not the other, so nothing was excluded **by name** —
the difference is five days of growth, plus `runs/` which is 4,000× smaller in the backup
and cannot be growth alone.

**The cost of a refresh, so the decision has numbers.**

| what a refresh would copy | now | in the 08-30 backup | delta |
|---|---|---|---|
| `data/` | 664 GB | 289.8 GB | +374 GB |
| checkpoints (403 files, pre-prune) | 317 GB | in `backup/` root | — |
| `runs/` | 24.5 GB | **5.6 MB** | **+24.5 GB, effectively all of it** |
| `bench_eff/` | 3.04 GB | 996 MB | +2.0 GB |
| repo (scripts/docs/facts/eval/datagen) | ~9 MB | ~1.8 MB | +7 MB |
| **total on disk today** | **≈ 1.0 TiB** | **365.2 GiB** | |

`/data01` has **2.67 TiB free**, so a full copy fits with room to spare, and an incremental
rsync transfers only the delta. **Open questions for the user, both theirs:**

1. Refresh it now, schedule it (daily? per milestone?), or let it age as an 08-30 reference
   point?
2. Should `runs/` be in scope? It is the audit trail and it is currently the one directory
   the backup does not really hold.

## 4. The host: nothing to clean, and that is a measurement

The cleanup order anticipated host-side leftovers — the 2026-09-03 `/tmp/absmoke` copy, `tn
push` landings, a stale `/work/aupai` host tree. **None of them exist.** Host `/tmp` totals
**1.46 GiB** across 5,234 files, the largest single item `qwen.tgz` at 0.76 GiB from
2026-08-28. `absmoke` was cleaned when it happened and left nothing.

**What a periodic host job may remove**, if one is ever built: our own files under host
`/tmp` older than 24 h, matched by a path prefix we own — nothing else. **What it must
never touch:** `/var/lib/kubelet` (that *is* `/work`, 927.6 GiB, live), `/var/lib/containerd`
(image layers for every pod on the node), `/root/podman` (187 GB, another workflow),
`/data0x` in any form, and any path not created by us.

Given the measurement, **the recommendation is to build no host job at all** until a host
leftover is observed twice. The one incident that motivated the order was a single `cp -a`
that a person cleaned the same day.

## 5. Scratch and rotation

- **Scratch root `/work/aupai/tmp/`** with a TTL sweep, so ad-hoc files have one home the
  sweeper knows. Today they land in the container's default cwd, `/sgl-workspace/sglang`,
  because a command that does not `cd` inherits it — which is also what made PR-11 misread
  307 processes as another project's.
- **`runs/*.log` rotation**: unbounded today. `ab_base_a_first.log` reached 3.8 MB on the
  pod while its committed snapshot held 8.4 KB — 0.2% of the run. Rotation must not
  silently truncate what a fact cites; rotate by move-and-compress, never in place.

## The populations these classes were measured on

**Every class below was swept by C1 (`ff035f77`) at ~05:5xZ, so a process table read after
that hour shows them empty.** db measured 37 live processes at ~07:4xZ and found (a) and
(c) with no instances and (b) with one — that is the sweep working, not classes that never
fired. The fixtures are the pre-sweep populations:

| class | instances before C1 | after |
|---|---|---|
| (a) pipe-stdout watcher | **307** | 0 |
| (b) `[ -f X ]` wait-loop | **3** | 0 |
| (c) tail on a finished run's log | **5** (6 reported; one was a grep self-match) | 0 |
| (d) ours and accounted | n/a — a protection, not a candidate | |
| (e) foreign / GPU-holding | 17 tileRL `tail -f /work/*.log` + 2 training chains | same |

**Build matchers against the recorded populations, not against a swept box**, and build no
matcher for a class with no recorded instance.

### One shape that resists both predicates, recorded rather than matched

```
until grep -q ALL-DONE runs/count_en_c4_both.log; do sleep 5; done    16.9 h, ppid 0
```

`runs/count_en_c4_both.log` is 355 bytes, mtime 2026-09-03T14:38Z, and **already contains
`ALL-DONE`** — the condition is satisfied and the loop is still spinning. It is caught by
neither the file-existence test (the file exists) nor a "target unwritten since the loop
started" test (it was written before, and writing is not the point). The likely cause is
the default-cwd trap again: its `cd` never took, so `runs/…` resolves elsewhere. **n=1 is
not a class; it stays `unclassified` and is recorded here so the next instance is
recognised.**

db then measured the mechanism rather than leaving it hypothesised: `/proc/3874083/cwd` is
**`/sgl-workspace/sglang`**, and `/sgl-workspace/sglang/runs` **does not exist at all**. The
loop's `cd` never took, so `runs/count_en_c4_both.log` resolves to a path that cannot
exist, and it polls forever while the real file has read `ALL-DONE` since 2026-09-03T14:38Z.

The general form — **a cwd-relative target that does not resolve from the process's own
`/proc/<pid>/cwd`** — is worth naming here because this container's default cwd makes it
common, and `AGENTS.md` already warns about it twice from other directions. **It is named,
not matched**: one instance does not justify a matcher, and db is not writing one.

## What this design does not solve

- **Zombies keep accumulating** until the user schedules the container change; forms 1–3 in
  §1 are the menu, and only form 2 is complete.
- **The sweeper cannot see the host** from inside, so every GPU and cgroup question is
  `unclassified` there — safe, but it means (e) protects rather than decides.
- **Nothing here refreshes the backup.** §3 is a decision, not a mechanism.
- **`runs/roster.json` has stale sockets** (db reports its own row wrong, and `44` and `b0`
  have been unreachable by name all session — every message to them went through 6e as a
  relay). That is a routing failure, not hygiene, and it belongs to that file's owner.
