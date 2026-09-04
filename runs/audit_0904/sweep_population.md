# `harness sweep`: the measured population, and what it does to the five classes

Container view (`~/bin/pod`, `/work/aupai`), 2026-09-04. Read before writing the classifier,
because three of the five classes as first specified match nothing on this box.

## The population

| | |
|---|---|
| **zombies** | **3,975** |
| live processes | 37 (34 `S`, 3 `R`) |
| `tail -f` older than 1h | 19 processes, 16 distinct targets |
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

## Class (c) is currently empty

All 19 `tail -f` processes watch `/work/*.log` — `wlad`, `rec`, `rec2`, `rec512`, `mcc`, `mcc2`,
`fus`, `fusk`, `amp`, `price512`, `widthjit`, `widthjit6`, `widthclean`, `scalemmlu`,
`scalefloor`. **Zero under `runs/`.** They are tileRL's, so the class as defined
(`tail` on a `runs/*.log` whose run has a closed experiments row) matches none of them, and a
looser "any old `tail -f`" would match 19 processes we do not own — class (e).

## Class (b) has one instance, and the parse must fit it

`until grep -q ALL-DONE runs/count_en_c4_both.log; do sleep 5; done`, ppid 1, 60,821s (16.9h).
It is a `grep -q` on a log, **not** `[ -f X ]`, so the file-existence parse both of us proposed
would not classify it. tilerl's mtime rule is the one that reaches it: the target has not been
written since the loop started, which is positive evidence the producer is gone.

## Class (a) has no instance to build a fixture from

None of the 37 live processes is a watcher with a pipe stdout. tilerl's a1/a2 definition is
sound; there is nothing here to demonstrate it against, and the doc should say "no instance
observed 2026-09-04" rather than have the code carry an unexercised matcher.

## Class (d) cannot protect anything yet

`runs/claims` is empty on the pod while lane jobs ran all day, because 10 of 11 GPU entry points
never call `card_claim` (`runs/audit_0904/card_claim_coverage.md`). So (d) rests on the
experiments row; a claim is corroboration, never the test. Adopted by tilerl.

## What the sweeper would actually do today

As specified: **one** sweepable process out of 4,012 — the 16.9h polling loop. The 3,975 zombies
it cannot touch are what anyone looking at this box would call the mess.

So the item as scoped buys one kill. What is worth building, in order of value measured here:

1. Report the zombie count and name the init fix. It is the finding, and it is one line of code.
2. Sweep the polling-loop shape, by tilerl's mtime rule — one real instance.
3. Leave (a), (c) and (e) as defined-but-unexercised, stated as such in the doc, and revisit when
   an instance exists.

Recorded rather than acted on: `--execute` defaults off, unclassified is never killed, and
`/proc/<pid>/cmdline` is re-read immediately before each signal and the pid abandoned if it no
longer matches — a pid is reused and a scan is minutes old by the time it acts.
