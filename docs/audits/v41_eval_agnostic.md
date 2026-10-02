---
question: Which eval-path places hard-code a V2 attention name, and do they block scoring a DeepSeek-V4.1 (CSA2) checkpoint with the HumanEval gate instrument?
status: retracted
source: grep over eval/ and scripts/loader.py at bf19a25d, 2026-09-10; PR #210 (docs/standards/v41_pivot.md, open). Retracted 2026-09-30 by 9588ee2b (scripts/loader.py, v41f/lm.py) and 665d7bb2 (the five raw decode loops).
---

# V4.1 eval-path architecture audit (66-2)

## RETRACTED 2026-09-30 (de)

**The verdict below does not hold for an `--arch v42` checkpoint, and the retraction is of
the verdict, not of the table.** The audit's own premise names its scope: "a V4.1 checkpoint
that is still a `HybridLM` with the same forward interface". v42 is neither. It is
`v41f/lm.py` `V42LM` with 5,139 state-dict keys (`embed.weight`,
`layers.0.attn.qproj.wq_a.weight`) against HybridLM's 35 (`tok.weight`,
`blocks.0.mixer.qkv.weight`), zero overlap, and until 9588ee2b its `forward` returned
`(hidden, hidden)` rather than logits.

What that voids, by sentence:

| voided claim | what was true at 2026-09-30 |
|---|---|
| "scores with **zero eval-side changes**" | `scripts/loader.py` constructed `HybridLM(cfg)` unconditionally, so every eval that goes through `load_checkpoint` could not score a v42 checkpoint at all. Fixed in 9588ee2b, which calls `train.build_model`. |
| "both evals then call `model(x)` and read logits" | They called `model(x)` and read a dim-1024 **hidden state**. It argmaxes to ids 0-1023, all valid in a 32,768-slot vocabulary, so nothing raised and the pass@1 was garbage. `V42LM.forward` now mirrors HybridLM's three-branch contract (9588ee2b), and `scripts/decode_guard.last_logits` refuses a non-vocab width at each of the five raw decode loops (665d7bb2). |
| "The gate instrument hard-codes no attention name" | Still true, and it was never the load-blocking property. Naming no attention class is not the same as building the class the checkpoint names — the audit tested the first and concluded the second. |

The **Places that DO name a V2 attention** table below stands as read at `bf19a25d` and is
kept for that reason; it was the question the audit actually answered. Nothing in it was
re-measured against v42, so treat every row as unmeasured for v42 rather than as carried
over.

The general form, for the next audit of this shape: an architecture-agnosticism claim is a
claim about the **construction** path and the **forward contract**, and a grep for class
names sees neither.


## Verdict (retracted for v42 by the section above; read as of bf19a25d)

The gate instrument hard-codes no attention name. `eval/humaneval_gen.py` and
`eval/humaneval_sample.py` call `scripts/loader.py:load_checkpoint`, which builds
`HybridLM(cfg)` from `ck["cfg"]` (loader.py:113) and nothing else; both evals then call
`model(x)` and read logits. `grep -n "csa|hca|attn_hybrid" eval/` is **0 matches**. A V4.1
checkpoint that is still a `HybridLM` with the same forward interface — which the pivot doc's
build order keeps ("rewrite CompressedSparseAttention … behind a csa2 flag", same container —
scores with **zero eval-side changes**, once `model.py` learns the csa2 branch. The sampled
instrument keeps both controls PR #207 added: the canonical_solution 164/164 control through
the sampled phase and the sample empty rate.

The csa2 branch itself is model-side work (pivot Step 1), not eval-path. Two model.py gates
decide whether a V4.1 checkpoint loads, named here because they sit on the load path:

- `model.py:1851` refuses a 0-KDA stack unless `rope_dims > 0`. Pivot Step 4 sets `rope_dims>0`,
  so the refusal lifts as specified; a V4.1 config that forgets it fails loudly, which is the
  intended behaviour.
- `model.py:1783` (`Block`) selects `GatedMLA`/`DeltaRecurrence` and has no csa2 branch yet.
  Until Step 1 lands, a V4.1 cfg constructs the wrong topology; after it, the eval path needs
  nothing.

## Places that DO name a V2 attention, with verdict

| place | what it names | verdict for V4.1 |
|---|---|---|
| `eval/humaneval_bpb.py:120-129` | refuses unless named layers' mixers are `GatedMLA` and `HAS_FA` | **Refuses by construction** on a CSA2 model. Off the gate path (gold-BPB likelihood eval, not pass@1). Needs a csa2-aware arm when V4.1 exists; do not "fix" by weakening the refusal — it is the prefix-cell instrument's only guard against scoring SDPA while reporting flash-attn. |
| `eval/run_eval.py:26` | `os.environ["FLA_FLASH_KDA"]="0"` unconditional | Inert: no `DeltaRecurrence` is constructed on a KDA-less model, so the kernel switch has no consumer. |
| `eval/code_zh.py:30`, `eval/base_matrix.py:38`, `eval/math_bpb.py:49`, `eval/humaneval_bpb.py:45`, `eval/math_zh.py:24` | `FLA_FLASH_KDA` setdefault | Same, inert. |
| `eval/nan_probe.py:66` | `HybridLM(Cfg)` from the LIVE Cfg | Training-side NaN probe; never loads a checkpoint. Correct as-is; it is not a scoring path. |
| `eval/domain_loss.py:341` | odd-length check justified by `chunk_kda`'s constraint | Conservative for a CSA2/SWA model (stricter than needed, never laxer); safe to leave until a V4.1 domain-loss run says otherwise. |
| `eval/prefix_mask.py`, `eval/loop_wrapper.py:182` | KDA/AttnRes-specific probe harnesses | Obsolete-for-V4.1 by construction, not blockers; they probe mechanisms V4.1 does not have. |

## One loader note

`load_checkpoint` backfills cfg keys missing from the checkpoint with live `Cfg` defaults
(loader.py:96-105). A V4.1 checkpoint that drops V2 keys (e.g. `attn_hybrid`) receives the
live default, which matches the `getattr(cfg, ..., False)` defaults the model code reads —
neutral by contract. The one pinned exception, `conv_doc_isolated=False` (loader.py:108),
predates this and stays correct.
