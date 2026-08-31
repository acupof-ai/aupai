---
question: Which failure classes did 2026-08-31 expose, and which check now closes each?
status: recorded
source: runs/retro.jsonl (tilerl, de; e1, 44, b0, 3b append later)
---

# Retro, 2026-08-31

Nine incidents from two sessions, on the day stage 1 launched. Four were caught before
they shipped. They fall into three classes; the first accounts for six of the nine and
already has a rule in AGENTS.md that names it.

Sessions that have not yet filed append to `runs/retro.jsonl`; this document is re-folded
then.

## Class 1 — a value or file used outside the regime that produced it

Six incidents. AGENTS.md already carries the rule: **Derived artifacts carry the
fingerprint of what produced them.** Each of these is that rule applied to something not
previously understood as a derived artifact — a rate, a token count, a quoted number.

| Incident | Owner | Caught? | What landed |
|---|---|---|---|
| 1.5-1.6 GiB/s reported as *the* cache read rate; it is one reader on an idle disk. Sizing the stage-1 gate from it gives 198 s for a startup that took 386 s — under the 600 s floor, so the derivation would have done nothing while looking correct | tilerl, caught by de | yes | `facts/efficiency.json#eff.cache_load_gates_startup` separates four rates and names the one that sizes the gate; `_CACHE_READ_GIBPS = 0.39` carries its derivation; `_selftest_gate_timeout` asserts against the measured 386 s |
| Token counts omitted the `<eos>` the training encoder appends: 4,020,618,525 against an independent 4,034,824,812 on math_owm, 0.35% | de | no | `scripts/count_tokens.py` is the single definition; `build_corpus` stamps `tokens_config` naming the convention |
| code_500 reported as 2.20%, read from `readout_30b.py`'s docstring example; the record says 0.0% | de | no | Corrected in `runs/tasks.jsonl` t39. `harness milestone` writes `runs/readout_<ckpt>.txt`; report from that file |
| Single-process re-tokenize started while another session held the cache with 8 encoders; throughput fell to 0.07 GiB/s and the job died in the 120 s gate | tilerl | no | `launch_30b.sh` names no gate; `harness launch` derives it from mix cache bytes |
| `harness task done` run on the pod, where `runs/tasks.jsonl` is a stale one-way copy | tilerl | no | Proposed: `harness task` refuses on the pod, the guard shape already at `harness.py:2491` |
| `splitlines()` used on JSONL; it breaks on U+2028/U+2029/U+0085, which `json.dumps` leaves raw. One document became four invalid fragments in the launch-path corpus | de | no | Rejoined; all 235 shards verified with strict `json.loads`; `malformed_lines: 0` in the stamp. Confirmed independently by a clean tokenize of 3,747,157 docs |

## Class 2 — a guard that cannot fire

Two incidents. Both looked like protection and provided none.

| Incident | Owner | Caught? | What landed |
|---|---|---|---|
| Manual-count ratchet written as `_MANUAL_BASELINE = len(_MANUAL_RULES)` — it moves with the set it pins | de | yes | Literal baseline, verified to FAIL on +1 (`d94fce6`) |
| Fallback pod-sync script whose delete logic would have removed live `train.py` and `harness.py` from a HEAD-derived list | tilerl | yes | Dropped; superseded by `pod_push.sh --all` |

Also in this class, though not an incident: two rule mappings claimed more than their
checks assert. `pinned_ids` checks that two special ids did not move, which catches a
rebuild after the fact but cannot see an unfreeze decision. Both demoted; the checked
count went 15 → 14.

## Class 3 — a test that writes to production

One incident.

| Incident | Owner | Caught? | What landed |
|---|---|---|---|
| The auto-resume selftest called `exp.py` with no root override and appended four real rows to `runs/experiments.jsonl`, two sharing an identity, which then failed the sync guard | de | no | `_supervise` and `_close_row` take an explicit root; the selftest passes a temp directory and suppresses the monitor |

`exp.py` takes no ambient root override by design — the ledger gets no environment
variable. A test that cannot redirect a writer writes to the real thing.

## Prose-only, no check

| Rule | Why no check |
|---|---|
| A deletion needs a per-file check for glob and runtime loaders | No static analysis sees a runtime glob. `vet_programs.py:37` globs `math_programs_l*_ext*.py`; a name scan reads 23 live generators as unreferenced, and a deletion pass nearly removed them. `reachability.py` now prints this above its own table |
| Quote a measurement only from the artifact that produced it | No artifact records where a number in a message came from |

## What changed in the harness

42 checks, each verified to FAIL on a broken world. Added on 2026-08-31:
`agents_rules_covered`, `curl_ipv4`, `no_foreground_pod_training`, and three selftests
(gate arithmetic against the measured startup, register union-merge, auto-resume across
its three exit paths).
