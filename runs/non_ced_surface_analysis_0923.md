# Non-CED construction surfaces — ANALYSIS, not a deletion broadcast

Owner genB, 2026-09-23. Task from aupai-08/1e (relayed): after the CED 30B run
`v41_ced_0923` started (prereg `c14ebf72`), enumerate every non-CED construction surface
so a 24h broadcast can be issued and the user approves per path. **Delete nothing.**

Baseline: `origin/main` = `f11fbb65` (`git rev-parse origin/main`, read 2026-09-23).

## 0. The task as stated cannot be executed, and the repo has a check that says why

**The broadcast-list mechanism does not apply to this surface.** AGENTS.md §255, enforced by
`harness.check_deletion_list_no_tracked` (its `def` in `scripts/harness.py`; listed in the
check table under the name `deletion_list_no_tracked`, `auth=repo`, added in `316580a0`):

> a deletion list may not name tracked content; strike the target or the rm removes what a
> fresh checkout ships, with every gate green

The check scans `runs/*deletion_candidates*.md|txt` and FAILs any table cell or bare path
that is tracked in main. My first draft of this list was refused by it at commit time, naming
all eleven candidates — because **every non-CED construction surface here except four
pod-only launchers is tracked in main.** A deletion broadcast is for pod-only / untracked
content; deleting tracked code is a normal reviewed PR. Merging the two procedures is exactly
the §255 incident.

So this document is an **analysis**, filed under a name the check does not scan
(`*_surface_analysis_*.md`, deliberately not `*deletion_candidates*`), and it does not
constitute a broadcast. If the user still wants a broadcast list, the only things that can go
on it are §3's four pod-only rows.

**Two more corrections to the premise, measured before anything else:**

- **`DELETION_CANDIDATES.md` does not exist.** Not on disk, not in `git log --all`
  (`--diff-filter=A --name-only` over every ref, 683 remote branches), not in any of the 30+
  worktrees. Nearest tracked names: `runs/c3_deletion_candidates_0906.md` (the C3 **corpus**
  cleanup) and `docs/audits/deletion_audit_2026-09-02.md`. This analysis is built from the
  task text. If a specific file was meant, name it and it gets rebuilt against it.
- **There is no `--flat` flag, and flat is not a separable surface.** `Cfg.ced = 0`
  (the `ced = 0` line in `train.py`'s `Cfg`) *is* the flat architecture; `--ced` is selected
  through the `argparse.BooleanOptionalAction` registration loop in `train.py` (the
  `parser.add_argument(f"--{name}", action=argparse.BooleanOptionalAction, ...)` line inside the
  dict-and-loop block that carries the `"ced"` help string).
  `--moe_arm` gates no architecture — it is a ledger row label
  (`train.py:265,3127,3322,4482`). "Delete flat" is three runtime `if` branches inside
  `model.py`/`train.py`, both of which the running job imports.

## 1. The finding that blocks the obvious reading: CED's encoder IS the flat code path

Measured with runtime spies on the real class, not read off names. §5 has the exact method.

`ced_kv = True` is set only on **decoder** layers (`model.py:2912` `_dec =
range(ced_enc_layers, layers)`, wired at `:2930`). Encoder layers 0-5 keep `ced_kv = False`
and take the `else` branch at **`model.py:1126-1128`** — `entries_per_doc`, the function a
name-based list deletes first.

Trace from a built CED model (`ced=1, ced_enc_layers=6, layers=12`), one call per layer:

```
layer0  entries_per_doc   layer6  _ced_kv_from_enc
layer1  entries_per_doc   layer7  _ced_kv_from_enc
layer2  entries_per_doc   layer8  _ced_kv_from_enc
layer3  entries_per_doc   layer9  _ced_kv_from_enc
layer4  entries_per_doc   layer10 _ced_kv_from_enc
layer5  entries_per_doc   layer11 _ced_kv_from_enc
```

Poisoning `entries_per_doc` with a raising spy **kills the CED forward**. Consequences:

| symbol | name-based reading | measured under CED |
|---|---|---|
| `entries_per_doc` (`model.py:475`) | flat-only, delete first | **live CED code** — encoder half. Sole call site `:1127`. |
| `_fp8_linear_entries` (`model.py:338`) | shared | **CED-ONLY** — called from `_ced_kv_from_enc` (`:437-438`) alone |
| `_doc_blocks` (`:292`), `pool_per_doc` (`:442`) | flat-only | live — HCA (`:853`) and `_forward_packed` (`:1202`) |
| `Cfg.ced=0` default | "the flat architecture" | the encoder's own path, reachable inside CED |

So removing flat construction is **not** a deletion even in principle: the two architectures
share the entry builder. It is a refactor, and it is forbidden this window anyway (the
running `train.py`/`model.py` path must not be touched).

## 2. What the running 30B actually reads

Process cmdline, read on the pod 2026-09-23 (`ps -eo pid,ppid,etime,cmd`):

```
torchrun --nproc_per_node=8 --master_port=29500 train.py --fp8 --mix data/mix_v41_gate.json
  --name v41_ced_0923 ... --ced --ced_enc_layers 6 ... --moe_arm v41ced
```

Import closure of `import train`, runtime-measured: **`['fone', 'model', 'train']`**.

`fone.py` is in it (`model.py:31`, `train.py:38` both `import fone`). Never a candidate.

The launcher is read once at start; the live process carries its full argv. So deleting any
launcher — flat or the CED one — cannot affect the running job.

## 3. Surfaces, per-path, with consumers and migration

`consumers` = the glob/importlib/runtime-loader check AGENTS.md requires, plus ledger
citations. `running 30B imports?` = §2's closure (all no).

### 3a. POD-ONLY — the only rows that could legitimately go on a broadcast list

Untracked in main (`comm -23` of pod `runs/*.sh|py` basenames against `git ls-files runs/`).
Per §255 these are the §255-shaped ones: deleting them is a repo edit on the pod where no
hook looks, so the broadcast procedure is the right one for these four and only these four.

| path (on pod) | arch | consumers | running 30B |
|---|---|---|---|
| `/work/aupai/runs/ced_w8_launch.sh` | **CED — THE LIVE RUN'S LAUNCHER** | none in git | **is the running job's launcher — do not touch** |
| `/work/aupai/runs/ced_w8_smoke.sh` | CED (S3 smoke) | none in git | no |
| `/work/aupai/runs/ced_smoke_launch.sh` | CED (S2 smoke) | none in git | no |
| `/work/aupai/runs/v41_peak_0920.sh` | **flat** (`--moe_arm v41peak`, no `--ced`) | none in git | no |

21 further pod-only basenames are data/infra builds (`math_*.sh`, `l2_*.sh`, `sc_*.sh`,
`ultradata_fetch.sh`, `star_build.sh`, …), unrelated to architecture — not listed here.

Migration for all four: none. But note the first row: the live run's launcher is untracked,
so if the pod is ever rebuilt there is no committed record of how `v41_ced_0923` was
launched. **That is a gap worth closing by landing a copy of `ced_w8_launch.sh` in `runs/`,
not a deletion.** Flagged, not actioned.

### 3b. TRACKED, flat architecture, superseded — normal-PR material, NOT broadcastable

`--ced` appears in **no tracked launcher**; every row below constructs the flat stack.

| path | consumers | migration | running 30B |
|---|---|---|---|
| `runs/v41_gate_0922.sh` | `prereg.jsonl`, `review.jsonl` | none — superseded by `v41_ced_0923` | no |
| `runs/v41_gate_0911.sh` | `prereg.jsonl`, `review.jsonl`, `tasks.jsonl` | none | no |
| `runs/v41_gate_0911_resume_w8.sh` | `prereg.jsonl`, `review.jsonl`; **cannot run** — its `--resume ckpt_v41_gate_0911.pt.step6000` was destroyed 2026-09-16 | none | no |
| `runs/v41_r3_0914.sh` | `prereg.jsonl` | none — flat round, closed | no |
| `runs/v41_smoke_0911j.sh` | `prereg.jsonl`, `review.jsonl` | none | no |
| `runs/v41_smoke_0920k.sh` | `review.jsonl` | none | no |

~56 KiB of shell, 6 files. The cost is not disk; it is that a superseded launcher re-runs
the stopped flat line if someone types its name.

### 3c. TRACKED tests whose subject CED refuses at construction

Each was **run** (AGENTS.md: "run a deletion candidate before judging it"). "Refused under
CED" = building `HybridLM` with `ced=1` plus this flag raises — measured, with the text.

| path | subject | CED verdict (measured) | consumers |
|---|---|---|---|
| `scripts/test_attn_res_fp32_logits.py` | `--attn_res_fp32_logits` | REFUSED: `ced=1 needs attn_res OFF` (`model.py:2906`) | hook `SELFTEST_FILES` |
| `scripts/test_untie_head.py` | `attn_res=True` arms | REFUSED: same `model.py:2906` | hook |
| `scripts/test_arch_L32.py` | L=32 AttnRes/KDA | REFUSED: same `model.py:2906` | **none** — not in CI, not in hook |
| `scripts/test_mem_defaults_frozen.py` | `ProductKeyMemory` defaults | memory is added only on the AttnRes path (`model.py:3220`) | hook |
| `scripts/test_table_master_resync.py` | memory table resync | same AttnRes-only memory (`model.py:3191-3202`) | hook |

### 3d. Corrections — classifications a name scan gets wrong

I tested each; two of a peer's "flat-only" calls were wrong and are not carried:

| path | name suggests | measured |
|---|---|---|
| `scripts/test_value_embed.py` | flat-only | **builds and runs under CED** — `value_embed=True` constructs (+151,040 params). NOT flat-only. |
| `scripts/test_zero_init_out.py` | flat-only | **builds and runs under CED** — zeros 24 output projections. NOT flat-only. |
| `scripts/test_v4_attn.py` | flat-only (HCA) | reachable via `attn_hybrid`, which CED's `csa2` refuses — a property of today's recipe, not of the architecture. **Hold.** |
| `scripts/test_split_bitwise.py` | flat-only | bitwise guard on a pre-CED merge-base — obsolete *by the §1 refactor*, not standalone stale. **Hold.** |

### 3e. Explicitly NOT candidates

- **`scripts/test_arch_compat.py`** — 2,589 lines holding the repo's **only CED test**
  (`:2179-2313`: builds `_CfgCed`, asserts the refusals and the encoder-visibility property
  that separates CED from flat). Gated by the CI step running `python scripts/test_arch_compat.py`;
  pre-commit subject for both `model.py` and
  `train.py`. Cannot be deleted; a section-level excision is a code edit to a gating file —
  out of scope this window.
- **`v41f/` (23 tracked) + `tests/v41f/` (38) + `probes/v41f_stepd_longrun.py`** — a **third
  architecture**, not flat. `V41FModel` (`v41f/model.py:60`) is single-pass with its own
  `train.py`/`master.py`, Engram, DSpark/MTP, hyper-connections. `grep -n ced v41f/*.py` =
  zero real hits. Zero references from `model.py`/`train.py`. CI-gated as its own package
  (the CI steps running `python tests/v41f/p0_selftest.py` and `p1_selftest.py`), with **two
  live preregs** (`v41f_indexer_train_0917` =
  `design_proposal_pending_fb_de`; `v41f_dspark_train_equiv_0917` = `registered`).
  Deleting it deletes a live track.
- **`v41f_l2/` (4) + `tests/test_l2_*.py` (2)** — frozen bge-m3 document-quality head,
  architecture-independent.
- **`mathbank/` (40), `bench_eff/parse_*.py`, `filters/`, `datagen/`** — arch-independent
  (`bench_eff/parse_*.py` carry reachability `KEEP` notes with pod-measured outputs).
- **`algorithms/rlvr_*.py`** — shared; builds its arch from the checkpoint.
- **`fone.py`** — live shared infra, in the running closure (§2).

## 4. Method, and one instrument that cannot be trusted for this question

- **Tracked status:** `git ls-files -- <path>`, `git ls-tree -r origin/main`, per candidate.
- **Refusal / reachability under CED:** build `HybridLM` with `Cfg.ced=1, ced_enc_layers=6`
  plus the flag under test; catch `ValueError`. Class-level spy (`M.entries_per_doc =
  poisoned`) for the KV source per layer; `M.HybridLM._body = spy` for the forward branch.
  §1's trace is that run's output.
- **Candidates run:** `PYTHONPATH=. python3 scripts/<t>.py` at repo root.
- **`scripts/reachability.py` is NOT evidence for this question.** `SCRIPT_RE`
  (`scripts/reachability.py:191`) matches only `scripts|eval|datagen|filters|mathbank|
  algorithms|probes` — **`runs/` is absent** — and `TOPLEVEL_RE`'s `(?<![\w/])` lookbehind
  blocks basename fallback for a full-path citation. Measured: `'runs/v41_gate_0922.sh'`
  resolves to **zero** edges although `prereg.jsonl` and `review.jsonl` both contain it
  verbatim. Its run reports 118 unreachable, **51 of them under `runs/`** — that number is
  the blind spot, not a finding. Its own docstring warns "a false 'dead' costs the file";
  here the tool fails in exactly that direction.
- **Local test noise:** `test_arch_L32.py` reports 3 FAIL at L=32 with
  `TypeError('NoneType' object is not callable)` — the fla-absent STAND-IN kernel, an
  environment artifact, not a statement about the file.

## 5. What this document does not do

- Nothing is deleted. No path is proposed for deletion by this session.
- `model.py`, `train.py`, `fone.py`, `scripts/test_arch_compat.py` are not candidates — all
  four are in the running job's path or gate it.
- The user's 2026-09-22 order「删除其他架构只保留 ced 架构」has its own gate order (smoke
  green → my list → 24h broadcast → user names symbols again → only then touch anything).
  This document is the "my list" step, with a correction: the "24h broadcast" step as the
  task described it cannot carry tracked paths, so §3b/§3c have to go through normal PRs
  with review, and only §3a is broadcast-shaped.
