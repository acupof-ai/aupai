# Flat-refactor sequence — draft PRs, none merges while `v41_ced_0923` runs

Owner genB, 2026-09-23. Companion to `runs/non_ced_surface_analysis_0923.md` (§1 established
that CED's encoder reuses the flat entry builder, so "remove flat" is a refactor).

Ruling this executes (aupai-08/1e, accepted): **no PR touching `model.py`/`train.py` or any
other file on the running job's import path merges while `v41_ced_0923` is running — a resume
after a crash would load the new code.** Draft PRs open now and wait for the run to end.
Files off the import path go through ordinary PRs now.

## The measured facts the split rests on

**Three body branches, orthogonal (measured with a `_body` spy, one config each):**

| config | branch taken |
|---|---|
| `ced=1 attn_res=0` — the live run | CED two-pass (`model.py:3161-3171`) |
| `ced=0 attn_res=0` | FLAT single-pass (`model.py:3172-3185`) |
| `ced=0 attn_res=1` — the legacy 30B line | AttnRes body (`model.py:3186-3226`) |
| `ced=1 attn_res=1` | REFUSED (`model.py:2906`) |

So "remove flat" removes branch 2 and nothing else. **AttnRes is a separate axis** and is not
part of this refactor; any step that drops it is a different decision with its own evidence.

**Line coverage under CED** (`sys.settrace` on `model.py`, live-run-shaped config):
`_body` executes exactly `[3149, 3161..3171]`; the flat single-pass `3172-3186` and the
AttnRes body `3188-3226` execute **0 lines**. In `_forward_csa2` both the `ced_kv` branch
(`1115-1125`, decoder, 3 lines hit) and the `else` (`1126-1128`, encoder, 2 lines hit) run.

**Byte-identical check is possible** (the acceptance for every step): building the same CED
config at the same seed twice gives the same `sha256` over the fp32 logits
(`732a951128074d73` both times, `mean|logit|` 0.18201710), and a different seed is
distinguishable (`0c0de90061532e82`). CPU, seconds per run.

**Entry builder stays.** `entries_per_doc` (`model.py:475`) is live CED code — it is the
encoder's global-KV source. A step that deletes it deletes the encoder. It gets a
CED-appropriate name or a clarifying docstring, not a deletion.

## The sequence

Each step is one draft PR, off fresh `origin/main`, and each declares: the known-answer test
(red on the broken version, green on the fixed one), the byte-identical CED forward check,
and the files touched. Steps 1-3 touch the import path, so they wait. Step 4 does not.

### Step 1 — cover the encoder branch (test only, no source change)

Add to `scripts/test_arch_compat.py` (or a new `scripts/test_ced_encoder_path.py`):
(1) a spy asserting `entries_per_doc` is entered once per encoder layer and never for a
decoder layer, and `_ced_kv_from_enc` the converse; (2) the byte-identical CED forward
digest. **Known-answer:** the spy goes red on a mutant that sets `ced_kv=True` on every
layer (the exact defect the CED/decoder wiring exists to prevent) and on a mutant that
drops the `else`. This step is what makes steps 2-3 falsifiable, so it lands first.

### Step 2 — WITHDRAWN (measured after writing it): the extraction is ceremony

I wrote this step to "isolate the flat single-pass branch behind one predicate" before
checking what it would buy. It buys nothing:

- `_body` has **one caller** (`model.py:3266`, `HybridLM.forward`), so "isolate it behind
  one call site" is already true.
- The branch is **14 lines** (`model.py:3172-3185`) and moves nowhere: it is read by all
  **six** flat launchers, every one of which passes `--no-attn_res` and no `--ced`
  (`v41_gate_0922.sh`, `v41_gate_0911.sh`, `v41_r3_0914.sh`, `v41_smoke_0920k.sh`,
  `v41_smoke_0911j.sh`, `v42_textbook_ab.sh` — checked per file). Extracting it into a method
  pays a diff to relocate code that step 3 then deletes.
- Its `pkg` slot logic is **duplicated** in the AttnRes branch below (`model.py:3186+` sets
  `pkg` the same way), so an extraction would have to either copy that too or leave the
  duplication — a refactor whose only output is a second copy.

Per the same reasoning that made step 1 a test rather than a source change: a step earns its
place by making something falsifiable or removing something. Step 2 does neither, so step 3
follows step 1 directly. Kept here rather than deleted so the next reader sees it was
considered and why it was dropped.

### Step 3 — delete the flat single-pass branch and the flat-only branches that follow

`model.py:3172-3185` goes; `_forward_csa2`'s `else` **stays** (encoder). The flat launchers
(§3b) and the tests whose subject CED refuses at construction (§3c) go in the same PR or a
sibling one, listed in the body. **Known-answer:** a mutant that restores the flat branch
must red; the CED digest must be unchanged.

### Step 4 — off-import-path files, ordinary PRs now

**ENUMERATED 2026-09-23, and the count is smaller than when this step was written. OPENED
ONLY AFTER #659 MERGES** (`runs/ced_w8_launch.sh` lands there; deleting the flat launchers
first would leave the repo with no tracked launcher for any architecture).

Per-file consumer check (the AGENTS deletion rule: grep glob/importlib over the directory,
plus ledger and AGENTS.md citations, plus whether anything actually INVOKES it):

| path | invocations | AGENTS.md | verdict |
|---|---|---|---|
| `runs/v41_gate_0922.sh` | 0 | 0 | deletable |
| `runs/v41_r3_0914.sh` | 0 | 0 | deletable |
| `runs/v41_smoke_0911j.sh` | 0 | 0 | deletable |
| `runs/v41_smoke_0920k.sh` | 0 | 0 | deletable |
| `runs/v41_gate_0911_resume_w8.sh` | 0 | 0 | deletable |
| `runs/v41_gate_0911.sh` | **2 tracked** | **0** | **HOLD, not deletable on today's evidence** — `AGENTS.md` no longer cites it (the CED pivot #659 `034166b4` rewrote §8 and the entry-point gate row to `runs/ced_w8_launch.sh`; measured 2026-09-23 at `390c80b1`: `git grep v41_gate_0911 -- AGENTS.md` = 0), so the reason this row gave is gone. But it is **not** down to historical documents: measured at `390c80b1`, 38 files cite the string and the *launcher* `.sh` is named by `README.md:79` (a live row: "The committed gate line is `runs/v41_gate_0911.sh`"), by `docs/standards/v41_pivot.md:121` (the gate-run recipe), and by `scripts/harness.py:10412` (the comment saying `check_corpus_filters_fp` must verify the mix this launcher selects — and that check reads `data/mix_v41_gate.json`, not the `.sh` itself). Only the `ckpt_v41_gate_0911.pt` name is history (`facts/*.json`, `EXPERIMENTS.md`, the `runs/*.jsonl` ledgers). So deletion needs the README row retargeted and the harness comment repointed first; open it as its own PR, not as part of a flat-only sweep |
| `scripts/launch_30b.sh` | 6 | 1 | **KEEP** — `docs/standards/launch_ready_guards.md` is entirely about `launch_30b.sh --dry`; it is a documented training entry point, not a stale launcher |
| `scripts/run_ab_speedrun.sh` | 8 | 0 | **KEEP** — 15 rows in `runs/experiments.jsonl`; live A/B infrastructure |
| `scripts/run_ablation.sh` | 3 | 0 | **KEEP** — cited by `docs/lessons/speedrun_techniques_audit.md` as the A/B shape |
| `scripts/run_pretrain.sh` | 4 | 0 | KEEP (same family; not separately measured) |
| `runs/v42_textbook_ab.sh` | — | 0 | **HOLD** — prereg `textbook_continuation_ab_0914` is still `open` |

All five `runs/` experiment rows are closed (`stopped`/`ok`/`error`), so none is live.

**The five §3c tests are NOT in this step.** Measured on `attn_res` per file:
`test_untie_head` (`c.attn_every, c.attn_res = 2, True`) and `test_arch_L32` (a four-way
assignment setting `attn_res = True`) DO set it and are AttnRes tests;
`test_attn_res_fp32_logits` imports `AttnRes` directly; `test_mem_defaults_frozen` and
`test_table_master_resync` build `ProductKeyMemory`, which the CED body never reaches. All
five belong to **step 5**. (A first grep used `attn_res\s*=\s*True`, which misses the
multi-assignment lines and reported all five as non-AttnRes — wrong, corrected here.)

### Step 5 — AttnRes (in scope, ruled 2026-09-23; a THIRD architecture axis)

Ruled in scope: the user's order is 删除其他架构只保留 ced 架构, and AttnRes belongs to the
0830v1 line already retired in CLAUDE.md, so it is "another architecture". Same method as
steps 1-3, plus the checkpoint evidence below. Touches the import path → draft until the run
ends.

**The load surfaces (three, enumerated from the source):**

| # | surface | what it does for AttnRes |
|---|---|---|
| 1 | `scripts/loader.py` `cfg = SimpleNamespace(**ck["cfg"])` | rebuilds the model from the ckpt's own cfg, so a ckpt with `attn_res=True` gets AttnRes; the `for _k in vars(Cfg): ... setattr(cfg, _k, getattr(Cfg, _k))` loop below it backfills any key the ckpt lacks from the live `Cfg` |
| 2 | `model.py` `HybridLM.load_state_dict` — the `if self.attn_res and not any(k.startswith("final_ar.") for k in sd):` branch | if the model has AttnRes and the ckpt has no `final_ar.` key, prints and disables AttnRes, then loads strict |
| 3 | `infer_local.py` `self.attn_res = getattr(cfg, "attn_res", False)` and its use below | a SECOND, independent AttnRes reimplementation (plus `AttnRes`, the `class AttnRes(nn.Module)` in `model.py`, and `scripts/test_arch_L32.py`) |

**Checkpoint evidence (read-only listing, requested before deleting the loader path):**

Every checkpoint reachable on the pod, via `scripts/ckpt_info.py` (mmap, read-only) and a
state_dict key scan — **13 files, and not one needs AttnRes:**

| checkpoint(s) | cfg.attn_res | AttnRes param keys | KDA keys |
|---|---|---|---|
| 12 files: `ckpt_v41_ced_smoke_0922.pt` (+`.ep1`), `ckpt_v41_ced_w8smoke_0923.pt` (+`.ep1`), `ckpt_v41_peak_0920.pt` (+`.ep1`), `ckpt_v41_smoke_0920k.pt` (+`.ep1`), `ckpt_v41_gate_0922.pt.{step4000,step6000,step8000,interrupt.step8196}` | `false` | 0 | 0 |
| `/data00/ckpt_k3-mla_2b_step2000.pt` | **ABSENT** | **0** | 0 (but `mixer.A_log`, `dt_bias`, `short_conv`, legacy `gate_proj`/`beta_proj` → KDA line) |

The K3 file is the only ckpt whose cfg lacks `attn_res`. the `for _k in vars(Cfg)` backfill loop in `scripts/loader.py` sets it to the
live `Cfg.attn_res = True`, so surface 2 is what saves it — and surface 2 works: its
state_dict has **zero** `final_ar.`/`ar1.`/`ar2.` keys (153 keys total: `tok`, `blocks`,
`norm`, `head`), so AttnRes is disabled at load. **Nothing on the pod exercises AttnRes;
surface 2 is the only thing that would, and only for that one K3 file.**

Two notes for the record:

- **No backup exists.** `/mnt/data02` does not exist on the pod (`mount | grep data02` empty,
  `ls /mnt/data02/aupai_backup/` → No such file or directory). So `AGENTS.md`'s pod-deletion
  backup step is currently unrunnable, and this evidence cannot be cross-checked against one.
- **A doc disagrees with the pod.** the `eff.fp8_nan_without_grad_ckpt_unreproduced` entry's `boundary` in `facts/efficiency.json` states the inputs
  `data/sft/sft_v3.pt` and `ckpt_k3-mla_2b_step2000.pt` "no longer exist on the pod". The K3
  checkpoint **does** exist there today (479,293,133 B, mtime 2026-08-25). Current disk wins;
  the fact's boundary sentence is stale. Reported, not corrected here.

**What step 5 must not break, and the coverage that already exists.** `CLAUDE.md` (a symlink to `AGENTS.md`) states, in the paragraph beginning
"AttnRes does not cross the future CED boundary",
the contract the AttnRes code is serving: "Old checkpoints still load via `_cfg`
(`scripts/loader.py`)". Two facts bound step 5:

- `scripts/test_arch_compat.py:145` **already asserts** the auto-disable
  (`assert new.attn_res is False and Cfg.attn_res is False, "old ckpt must disable AttnRes"`),
  on a synthetic legacy state_dict with no `final_ar.` keys. So step 5 must keep that
  assertion's *effect*: whatever replaces surface 2 has to produce "this ckpt is AttnRes-era,
  here is what happens" rather than a strict-load traceback.
- `Cfg.attn_res` still defaults **True** (the `attn_res = True` field in `train.py`'s `Cfg`, since `b3cad874` "arch(0830v1): full
  causal MLA, AttnRes on by default"). With flat gone the flag still has a job: it is not
  refused under `ced=0`, so `Cfg.ced = 1` becomes the thing that selects CED and `attn_res`
  becomes dead-by-configuration rather than dead-by-code. Flipping the default is a **separate
  decision** with its own blast radius (it changes the default architecture for any launcher
  that passes neither flag) and is NOT part of step 5.

So step 5's known-answer test is: a legacy AttnRes-era state_dict still loads, by whichever
mechanism replaces surface 2, and `scripts/test_arch_compat.py:145`'s assertion is either kept
or replaced by a strictly stronger one. Deleting surface 2 with nothing in its place reds that
line — which is the mutant this step is accepted against.

## Explicitly not scheduled

- **AttnRes** — moved INTO scope as step 5 (above).
- **`entries_per_doc` deletion** — it is the encoder. Name/docstring only.
- **HCA removal** — reachable under `attn_hybrid`; a recipe decision, not a refactor step.
- **`v41f/`** — third architecture with its own live preregs.
- **`test_arch_compat.py` deletion** — it holds the repo's only CED test and gates both files.

## Preconditions before step 2 merges

1. `v41_ced_0923` has ended (not running, not resumable mid-flight).
2. Step 1 is merged and its spy is red on the mutant.
3. The CED forward digest is recorded in the PR body, computed at a stated seed and shape.
