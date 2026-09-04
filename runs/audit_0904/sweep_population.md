# `harness sweep`: the measured population, and what it does to the five classes

Container view (`~/bin/pod`, `/work/aupai`), 2026-09-04.

> **CORRECTION, and it invalidates this document's first conclusion.** This was measured
> **after** tilerl's C1 sweep (`ff035f77`, 13:35 +0800) killed exactly the populations I then
> reported as absent — verified in that commit: **306 orphaned `tail -F runs/events.jsonl`**
> (class a), **3 `until [ -f data/code_supply/measure_*.json ]` loops** (class b), **5 stale
> `tail -f runs/*.log`** (class c). My "class (a) has no instance", "class (c) is empty" and
> "one sweepable process out of 4,012" were the sweep working, roughly 90 minutes earlier, read
> as classes that never had instances.
>
> This is the repo's own named defect — absence measured on a mutated world and reported as a
> property of the class. I made it while writing a document about not doing that. The matchers
> are to be built against the pre-sweep populations recorded in `ff035f77` and PR-11, not
> against the table below. Everything below stands as *post-C1* state and nothing more.

## The population (POST-C1 — not the state the classes should be designed against)

| | |
|---|---|
| **zombies** | **3,975** |
| live processes | 37 (34 `S`, 3 `R`) |
| `tail -f` older than 1h | 19 processes, 16 distinct targets, all tileRL's |
| `until`/`while` polling loops | 4 |
| live processes with `ppid 0` | 21 (`ppid 0` here is the `crictl exec` entry point, not init) |
| tileRL training chains | 2, on cards 1 and 3 (`grpo-gsm8k-27b`, `opd-gsm8k-27b`) |

**PID 1 is `sleep infinity`.** It reaps nothing, so every exited child accumulates. 3,975
zombies against 37 live processes is the environment's actual condition, and no sweeper can
change it: a zombie has already exited, `kill` does nothing, and only PID 1 reaping clears it.
The fix is a real init (`--init`, or tini as PID 1), not `harness sweep`.

This is not academic — it has already cost time twice. A zombie keeps its `/proc` entry, so
`[ -d /proc/<pid> ]` reads true for a dead process: fb's scoring chain waited 31 minutes on a
reaped pid with card 6 idle, and `kill -0` plus `ps -p` both report a zombie as present, which is
why the standing rule is to read `ps -o stat=` and look for `Z`.

## Class (e): the cgroup test has zero discriminating power from inside

tilerl proposed replacing `FOREIGN_MARKERS` prose matching with a positive test: skip a process
whose `/proc/<hostpid>/cgroup` names a container other than ours. The prose criticism is right —
`card_claim.foreign_cards` decides foreignness by matching human-written strings in a JSON note,
and it called card 7 "another container" for a job that was in ours from another repository.

But the replacement does not work where the sweeper runs. Measured:

```
cgroup of PID 1                     kubepods/besteffort/pod95a05e32.../827d3e58...
cgroup of the tileRL grpo trainer   kubepods/besteffort/pod95a05e32.../827d3e58...  (identical)
cgroup of the tileRL opd trainer    identical
cgroup of an 82,500s `tail -f`      identical
distinct cgroups across all 37 live processes:  1
```

**One.** The tileRL trainers we must never touch are byte-identical in cgroup to our own
processes, because they run in our container from another repository. Repository, container and
namespace are three properties and the cgroup only sees the second.

And the namespace half holds from the third direction too: `nvidia-smi` inside the container
reports pids `3487226` and `3547785` (42.4 GB and 40.9 GB), and **neither resolves in container
`/proc`**. So an "is this a GPU holder" test built on intersecting the two pid sets marks
everything foreign — the same trap `check_card_held_without_claim` already documents.

Consequence: under tilerl's own rule that an unresolvable process is *unclassified, never
sweepable*, a container-side sweeper classifies **every** process as unclassified. Class (e) is
sound only if the sweeper reads host `/proc` through `tn exec`, which is a different tool in a
different filesystem view.

## Class (c): empty post-C1, not empty as a class

All 19 surviving `tail -f` watch `/work/*.log` — `wlad`, `rec`, `rec2`, `rec512`, `mcc`, `mcc2`,
`fus`, `fusk`, `amp`, `price512`, `widthjit`, `widthjit6`, `widthclean`, `scalemmlu`,
`scalefloor`. Zero under `runs/`. **That is the sweep's result, not the class's nature:** C1 killed
five `tail -f runs/*.log` this morning, so the aupai ones are gone and tileRL's remain — which is
correct, they are class (e). Build the matcher against C1's five.

## Class (b): the file-existence shape had 3 instances; the surviving one is a NEW shape

C1 killed three `until [ -f data/code_supply/measure_*.json ]` loops — the shape both tilerl and I
specified. The one still running is different, and it defeats both our predicates:

```
until grep -q ALL-DONE runs/count_en_c4_both.log; do sleep 5; done    16.9h, ppid 0
```

MEASURED, and the mechanism is now established rather than suspected:

| | |
|---|---|
| `/work/aupai/runs/count_en_c4_both.log` | 355 bytes, mtime Sep 3 14:38, **contains `ALL-DONE`** (count 1, and it is the last line) |
| the loop's `/proc/3874083/cwd` | **`/sgl-workspace/sglang`** — the container's default cwd |
| `/sgl-workspace/sglang/runs/` | **does not exist** |

So the condition is satisfied against the real file and the loop cannot see it: its `cd` never
took, `runs/…` resolves against `/sgl-workspace/sglang`, and it will poll forever. tilerl
hypothesised this; the `cwd` readlink confirms it. It is the same default-cwd trap that
`podput`'s relative-path footgun and the 307-watcher misclassification came from, and AGENTS.md
already states the cause: `~/bin/pod` lands in `/sgl-workspace/sglang`, so a command must `cd`
itself, and a `cd` inside a backgrounded chain does not reach what follows the `&`.

Neither predicate catches it — not file-existence (the real file exists), not
"target unwritten since the loop started" (it was written before, and writing is not the point).
**n=1 is not a class:** it stays unclassified, and the doc records the shape and why it resists a
predicate. A matcher that COULD catch it — cwd-relative target that does not resolve from the
process's own `cwd` — is worth noting as the general form, since the trap is common in this
environment even if this instance is single.

## Class (a): 306 instances this morning, none now

C1 killed 306 of 307 orphaned `tail -F runs/events.jsonl` (one skipped because its cmdline had
changed between scan and signal — the guard working). Nothing here now is a pipe-stdout watcher,
so the fixture comes from that recorded population, not from this box.

## Class (d) cannot protect anything yet

`runs/claims` is empty on the pod while lane jobs ran all day, because 10 of 11 GPU entry points
never call `card_claim` (`runs/audit_0904/card_claim_coverage.md`). So (d) rests on the
experiments row; a claim is corroboration, never the test. Adopted by tilerl.

## What the sweeper is worth

My first version concluded "one sweepable process out of 4,012, so the item buys one kill." That
was wrong for the reason at the top: I counted after the sweep. The correct count is **314 killed
by hand this morning** (306 + 3 + 5), and those classes accumulate — 307 watchers built up over
24 h once and will again. The value is not this box this hour.

Build, against the pre-sweep populations in `ff035f77`:

1. Class (a), from the 306 recorded watchers.
2. Class (b) file-existence shape, from the 3 recorded loops.
3. Class (c), from the 5 recorded `tail -f runs/*.log`.

Do not build: a matcher for the surviving `grep -q` loop (n=1, unclassified by design), or for
any class with no recorded instance.

Report, never sweep: the zombie count, with the init fix named. The reaping fix costs a container
recreation, which kills the user's tileRL jobs, so the user schedules it.

Standing guards, agreed with tilerl: `--execute` defaults off; unclassified is never killed;
`/proc/<pid>/cmdline` is re-read immediately before each signal and the pid abandoned if it no
longer matches — C1's one skip out of 307 is that guard working; `runs/sweeper.jsonl` records
killed **and** skipped with the reason, and the selector beside every count (PR-11 said 153
rolling checkpoints where b0's glob said 149, and the four-file gap was pure selector).
