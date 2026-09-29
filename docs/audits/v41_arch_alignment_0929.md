---
question: Does the running CED model (v41_ced_0923 / stage-2 v41_ced_0926) implement every open/deferred row of v41_pivot.md's "Config cross-check" table, and is CED Eq.1 faithful line by line?
status: measured
source: docs/standards/v41_pivot.md:29-57; docs/audits/deepseek_v41_flash_config.json; model.py / train.py read 2026-09-29; meta-device construction of the gate config on the pod; ckpt_v41_ced_0923.pt and ckpt_v41_ced_0926.pt state_dict + cfg
---

# V4.1 architecture alignment audit (2026-09-29)

Read-only. No model code changed, no card used. Verdicts compare the gate launch
(`runs/ced_w8_launch.sh`, CED 6/6) and the two trained checkpoints against the released
`DeepSeek-V4.1-Flash` config. Two independent witnesses establish what actually trained:
a meta-device build of the exact launch flags, and the saved checkpoints' state_dict keys
and cfg. They agree on every point below.

## Cross-check rows

| field (released value) | what the code/checkpoint actually is | verdict | evidence |
|---|---|---|---|
| `n_swa_only_layers` — table row says **match** (released `compress_ratios[0:2]=0`: first 2 layers SWA-only) | **0 SWA-only layers.** All 12 layers are `CompressedSparseAttention`; there is not one `PureSWA` parameter in either checkpoint. `--n_swa_only_layers 2` is a no-op in the gate because the per-layer kind/mode map is built only inside `if _hyb:` and the launch never sets `--attn_hybrid` (Cfg default False); with hybrid off `kind={}` and every GatedMLA reads the global `csa=True`. Meta build: `SUMMARY swa_only=0 csa2_full=6 ced_decoder=6`; checkpoints: 0 keys contain `.swa.`, layer 0 mixer submodules are `csa,kv_down,kv_up,o,qg`. | **不采纳 (latent, not chosen).** The table's "match" is wrong: the flag is accepted and recorded but inert. Two layers carry the global CSA2 entry branch + indexer the released model reserves for layer 2+. Aligning needs `--attn_hybrid` (which then also activates the `csa2_modes` map) — a topology change, not a flag flip on the same graph, so it cannot be applied to a live run. | `model.py:2905` `_hyb`; `model.py:2912-2947` map is hybrid-only; `model.py:1534-1535` PureSWA only under per-layer `cfg.swa`; launch omits hybrid (`runs/ced_w8_launch.sh:32-33`); witness: meta build + both ckpts have no `.swa.` keys |
| `kv_source_layer_ids=[2,8,14,20]` (Full refreshes KV every 6 layers) | No periodic Full refresh. Layers 0-5 are the encoder (each its own learned entries); layers 6-11 are decoders whose global KV is projected from the single boundary state `H_6` by that layer's own `W_KV/W_Z`. The flat `csa2_modes` F/R package thread is the mechanism that would host refreshes, but under the CED+non-hybrid gate `csa2_modes` is inert (`csa2_modes={}` at build) and CED installs its own branch instead. | **已对齐 to CED by design (偏离 literal field).** CED Eq.1 deliberately replaces the 40-layer F-refresh schedule with one encoder boundary. This is the user-ordered pivot, not drift. Re-adopting a layer-8 Full refresh would be a different architecture. | projection `model.py:390-438`; per-layer unshared pair `model.py:2997-3011`; two-pass boundary `model.py:3238-3248`; inert flat map `model.py:2912` |
| `compress_ratios` = 0,0 / 2×18 (enc) / 1×20 (dec) / 0,0,0 | Not a per-layer list anywhere. A single global `csa2_m=8` for every CSA2 layer; CED decoder entries are an 8-token-per-entry mean pool of `H_6` before the d→d projection. The released ratio compresses the **entry stride** (block_size×ratio: encoder 16 tokens/entry, decoder 8); the entry head width is unchanged on both sides. Ours is 8 tokens/entry in both halves. `CSA2Package` carries K/V tensors, not a ratio. | **不采纳.** Uniform 8:1 token stride; no encoder/decoder ratio distinction. Under CED the decoder global entries come from `H_6` regardless, so the released 16-vs-8 split does not map onto this graph. Change size = one stride constant per half if ever wanted; effect on the global-entry count is unmeasured. | `train.py:399` `csa2_m=8`; pool at fixed m `model.py:432-438`; no `compress_ratios` symbol in model.py/train.py (grep empty); released list `docs/audits/deepseek_v41_flash_config.json` |
| router `scoring_func=sqrtsoftplus`, `topk_method=noaux_tc`, `norm_topk_prob=true`, `routed_scaling_factor=1.5` | Affinity is plain `softmax` or `sigmoid(z)` selected by `router_score`; selection adds a selection-only fp32 `expert_bias` (aux-loss-free load balance); selected weights are renormalized within top-k. No softplus, no noaux_tc bias-correction term, and **no 1.5 output scaling**. ckpt 0923 has no `router_score` field (resolves to the softmax default); stage-2 ckpt 0926 records `router_score='sigmoid'`. Neither is sqrtsoftplus. | **下一轮预训练采纳 — small change, low numeric risk.** sqrtsoftplus is a one-line affinity change plus a 1.5 multiply on routed output; the bias/renorm machinery already exists. Must be set at a fresh start (it changes routing distribution from step 0; bolting onto a trained router is an untested perturbation). | score branch `model.py:2460-2466`; no scaling on the dispatch path `model.py:2598-2630`; default `train.py:266`; ckpts 0923(absent)/0926(sigmoid) |
| `hidden_act=silu`, `swiglu_limit=10.0` (clamped SwiGLU) | K3 **SiTU-GLU**: `β1·tanh(a/β1)·sigmoid(b)` then `β2·tanh(w2(gate)/β2)`, β1=4, β2=25, in both the dense FFN and every MoE expert. Bounded by tanh, not by a hard 10.0 clip; the inner gate is also bounded, which clamped SwiGLU does not do. | **不采纳 (recorded decision).** pivot "Decisions" says keep SiTU-GLU and add the clamp only behind an A/B. Both saturate large activations but are different functions near zero and different gradients; no A/B has run. | dense `model.py:1637-1649`; experts share it `model.py:2305-2307`, called `model.py:2600,2629`; decision `docs/standards/v41_pivot.md:87` |
| `rms_norm_eps=1e-20` | `1e-6`, hardcoded as the `RMSNorm.__init__` default; no Cfg field. Applied identically at pre-norm n1/n2 and final norm. | **下一轮预训练采纳 — trivial code change, new-run only.** Making eps a Cfg field and setting 1e-20 is a few lines, but it changes every layer's normalization numerics, so it takes effect only from a fresh init; on bf16 the practical difference is only seen when `mean(x²)` approaches 1e-6, far above normal activations. | `model.py:80` default; call sites `model.py:2720,2736,3011` |
| `tie_word_embeddings=false` | **Tied.** `untie_head=False`; the constructor aliases `head.weight = tok.weight`, and both checkpoints' head tensor is the same object as `tok.weight` (32768×1024). | **不采纳 for the current run (already trained tied); open for the next.** Untying adds +33.55M params and, more importantly, frees the LM-head lr from the embedding group's 0.1 — the code comment names that lr coupling as the active candidate. Requires a fresh run. | `train.py:449`; alias `model.py:3041-3042`; checkpoint head-is-tok data_ptr equality |
| `num_nextn_predict_layers=3` (MTP) | No MTP module, no extra predict layers. | **不采纳 — struck for the gate run by decision.** Scope is the single 30B→+10B tower; MTP is a loss/head addition for a later round. | zero `mtp/nextn/predict_layer` symbols in model.py/train.py; decision `docs/standards/v41_pivot.md:55` |
| `index_topk=512` (released; 4096 tokens/8 per entry at seq up to 1M) | `csa2_top_k=64` at seq 4096, m=8 → pool is only 512 entries, so 64 selects **1/8** of all entries (512 would select the entire pool = dense). Attended keys/query = 64·8 + 128 window = 640 of 4096 (15.6%). | **已对齐 in intent, re-chosen in number — do not copy 512.** The released ratio 512/~131k entries (seq 1M) is ~0.004 and cannot be held at seq 4096 without dense global attention. The constant was deliberately re-derived for 4096; fb's sizing range was 64-128. | `train.py:400`; rule/arithmetic `scripts/v41_size.py:112-119` (`entry_selectivity=64/512`, `attended_fraction=640/4096`) |

## CED Eq.1 — faithful or not

Eq.1 / pivot §Build-order Step 5: each decoder layer projects global entries `C^l` and
compression weights `Z^l` from the encoder boundary state `H_{L/2}` through its own
unshared d→d pair; mask causal everywhere.

| requirement | implemented | evidence |
|---|---|---|
| split 6 encode / 6 decode at `H_6` | yes — `ced_enc_layers=6` validated at construction; encoder pass runs blocks 0-5, `h_enc=x`, decoder blocks 6-11 read the stashed state | `model.py:2968-3011`, `model.py:3238-3248` |
| decoder global KV/Z projected **from `H_6`**, not from the layer's own K/V | yes — on `ced_kv`, `_ced_kv_from_enc(q, self._h_enc, ...)` runs and `entries_per_doc` (the layer's own learned reducer) is skipped; a missing stash raises | `model.py:1133-1143`, `model.py:390-438` |
| each decoder layer owns an **unshared** `W_KV`/`W_Z` | yes — one `nn.Linear(d,d)` per decoder CSA module at construction; checkpoint abs-sums differ across layers 6-11 in both runs | `model.py:2997-3011`; measured distinct w_kv in ckpts 0923 and 0926 |
| both project the **same** pooled `H_6` | yes — one `hb` mean tensor feeds `w_kv` (keys C) and `w_z` (values Z) | `model.py:432-438` |
| causal mask on the global entries | yes — entry visibility is `blk_last <= t` within the same document; window branch is the causal SWA mask | `_doc_blocks` visibility `model.py:292,329-331` (blk_last≤t, same doc); causal SWA window `model.py:1104` |
| teacher-forced, no prefill skip, two-pass gradient to encoder | yes — full teacher-forced two-pass body; `h_enc` is a live tensor (gradient not cut), checkpoint(b) on both halves | `model.py:3234-3248` |

Two deviations from a literal reading of the paper, both already accepted by the pivot:

1. **Mean pooling, then project.** The code mean-reduces each m-token document block of
   `H_6` before applying `W_KV/W_Z`; it does not learn the m>1 combine. The pivot names no
   reducer for CED and the docstring records this as a deliberate choice because CSA2's
   combine weights are unpublished (`model.py:410-413`).
2. **No periodic source refresh.** All six decoders read the same `H_6`; there is no
   Full re-projection at layer 8. That is the single-boundary CED definition the pivot
   chose over the flat stack; the 64-entry indexer still runs per decoder layer on top of
   the projected entries.

## What is most misaligned, ranked

1. **SWA-only first-two-layers is absent** despite the table's "match" — the only row where
   the audit contradicts the table rather than the released config. `n_swa_only_layers=2`
   silently does nothing without `--attn_hybrid`. Worth an explicit ruling: either align by
   enabling hybrid + a corrected mode map for a future run, or record that the gate
   intentionally runs CSA2 in all 12 layers and strike the "match".
2. Router scoring/scale (sigmoid/softmax + balance bias vs sqrtsoftplus + 1.5).
3. Untied LM head.
4. rms_norm_eps 1e-6 vs 1e-20.

Items 2-4 are small, local, fresh-run changes. Item 1 and the F-refresh/compress-ratio
rows are topology that the CED pivot intentionally traded away, not oversights.
