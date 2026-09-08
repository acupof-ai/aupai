# Controller board (fb) — 2026-09-08T08:5xZ

User order, this supersedes every plan below it: hand the cards back, then clean the whole codebase until there is not one redundant word or mark left. Nothing else runs until that is signed off.

## Cards

All eight released. No training, no eval, no probe, no launch. `runs/card_assignment.json` says so on every card. b0's `domain_bpb` re-score is the only GPU job queued and it waits with the rest.

## Cleanup program — one track per session

Scope is the whole tree: 473 Python files, 193,890 lines, 113 docs, 368 tracked files under `runs/`, 14 fact files, AGENTS.md at 439 lines. Every track ends in a PR reviewed by its pair.

| track | owner | scope | acceptance test |
|---|---|---|---|
| Scripts and entry points | b0 | `scripts/` 270 files, root entry points | every surviving file is reached from an entry point or a check; `scripts/reachability.py` clean; each candidate ran once before it was judged, with a per-file grep for `glob`/`importlib` |
| AGENTS.md | de | 439 lines | every rule maps to a check or a stated reason none can; `agents_rules_covered` green; no count in prose that a check already regenerates |
| Docs | 44 | `docs/` 113 files | frontmatter on every file; every `facts/` and `runs/prereg.jsonl` citation resolves and is current; no two documents answering the same question; `docs_root_clean` |
| Facts | e1 | `facts/` 14 files | `facts_well_formed` and `ckpt_facts_sources_present` green with zero WARN; every `retracted_value` list matches its entry; no source naming an absent checkpoint |
| Ledgers | 3b | `runs/` 368 tracked, 43 of them `.py`/`.sh` | no script under `runs/` unless a doc cites it; one schema per ledger; no stale `running` row; the 77 unregistered `.py` on the pod resolved |
| Eval and filters | tilerl-0a | `eval/` 54, `filters/` 4, `probes/` 13 | every metric divides by exactly what it scored; every metric has a known-answer test that CALLS the shipped function rather than reimplementing it |
| Data generation | 98 | `datagen/` 87, `mathbank/` 40 | no duplicate generator; the `vet_programs.py` glob registry reaches every live generator and nothing dead; then one index page saying where everything lives |

Deletion rule for every track: propose the list, run each candidate before judging it, the owner confirms each file by name, and the removal lands in a reviewed PR. The standing "no deletion without a named target" order is satisfied by the owner naming each file, not by skipping the step.

## What is finished and stays finished

| item | state |
|---|---|
| 30B leg | CLOSED at step 34,000 of 38,146 by user ruling. Recorded as an incomplete schedule, not an advance. Third on HumanEval gold bpb per task at 0.5609, behind `ckpt_0.2b_8b_b192` 0.5559 and its own annealed 8B sibling 0.5590; below two dense models on minimal pairs at 0.7653 against 0.8014; only clear win LAMBADA-en 0.3221 |
| What the anneal was worth | −6.89% unweighted mean `domain_bpb`, same run and same held-out rows: `ckpt_..._8b.pt.step9000` 0.35897 against `ckpt_..._8b.pt` 0.33424, all nine domains down, −4.0% to −12.4%. All nine token caches are stamped 2026-09-05, before both scorings, so the rows did not move |
| `domain_bpb` divisor defect | real, ~2×, known-answer test 8.000 true against 5.460 reported. Eight rows retracted in place. Fix is PR #79 with 3b. No rescaled level published; the prereg bar 0.334243 gets re-measured, never multiplied |
| Val prefix | latent defect, no observed instance. `train.py:2204` shuffles the whole document list before packing, so a pool that grows re-draws the held-out set: 50 added documents give 16–19% overlap, and the 5,000 cap holds the count, not the membership |
| `answer_present` at 3 demos | retired as a primary readout. Within one recipe it spans 0.1147–0.5433 across four checkpoints, sd 0.1852 against binomial 0.0201, so it cannot resolve a 1.42× effect. Both restoration arms are dead |
| v2 spec and prereg | merged, on main and on the pod. Amendment 1 carries the two CSA divergences from the DeepSeek-V4 reference and the 1.21× gate note |
| CSA attention | merged at 8c2308b7. Flag off is exact equality at 15,360 parameters; flag on differs at 0.154; 16 positions perturbed in k and v with no leaks; three mutants red |
| Repo and pod | main, origin/main and the pod stamp all at 8eecbd36. 738 files in sync, no refusing line. CI green on b134b88d |
| Pod disk | 95%, 109 GB free. Nothing deleted |

## Open user decisions

1. 30B composition for the next full run, once the cleanup is signed off.
2. Pod disk at 95%.
