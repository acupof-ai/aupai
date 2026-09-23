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
| `ced=0 attn_res=0` | FLAT single-pass (`model.py:3172-3186`) |
| `ced=0 attn_res=1` — the legacy 30B line | AttnRes body (`model.py:3188-3226`) |
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

### Step 2 — isolate the flat single-pass branch behind one predicate

Extract `model.py:3172-3186` into `_body_flat(self, x, cu)` and call it from one place.
No behaviour change; the byte-identical digest must not move. **Known-answer:** a mutant
that makes the new predicate always false reds every existing flat-config test.

### Step 3 — delete the flat single-pass branch and the flat-only branches that follow

`model.py:3172-3186` goes; `_forward_csa2`'s `else` **stays** (encoder). The flat launchers
(§3b) and the tests whose subject CED refuses at construction (§3c) go in the same PR or a
sibling one, listed in the body. **Known-answer:** a mutant that restores the flat branch
must red; the CED digest must be unchanged.

### Step 4 — off-import-path files, ordinary PRs now

Flat-only launchers (§3b, 6 files) and the five §3c tests. Ordinary PRs, review, no wait.
These touch nothing the running job imports.

## Explicitly not scheduled

- **AttnRes removal** — separate axis, separate evidence, not implied by this sequence.
- **`entries_per_doc` deletion** — it is the encoder. Name/docstring only.
- **HCA removal** — reachable under `attn_hybrid`; a recipe decision, not a refactor step.
- **`v41f/`** — third architecture with its own live preregs.
- **`test_arch_compat.py` deletion** — it holds the repo's only CED test and gates both files.

## Preconditions before step 2 merges

1. `v41_ced_0923` has ended (not running, not resumable mid-flight).
2. Step 1 is merged and its spy is red on the mutant.
3. The CED forward digest is recorded in the PR body, computed at a stated seed and shape.
