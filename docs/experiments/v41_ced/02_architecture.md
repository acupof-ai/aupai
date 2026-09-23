---
question: What is the CED architecture the gate run trains, what did the flat line it replaced look like, and why was flat stopped?
status: measured
source: runs/prereg.jsonl; facts/v41.json; runs/experiments.jsonl; model.py; pod training logs
---

# 02 — Flat line, pivot, CED architecture

## The flat line

Two flat gate attempts ran before CED.

| run | shape | outcome |
|---|---|---|
| v41_gate_0911 | flat CSA2+SWA, world 6 B4/accum8 steps 0–6000 then world 8 B4/accum6 to 17,411 | stopped by user order 2026-09-12 at 13.69B tokens; pod and checkpoints lost in the 2026-09-16 pod destruction |
| v41_gate_0922 | same model, fresh world-8 start | stopped by user order 2026-09-22 at step ~8,196/38,146, val 2.001, 0 NaN, peak 43.24 GiB, 27K tok/s/gpu |

Numbers: facts/v41.json#v41.gate_run_v41_gate_0911_summary for 0911 (val 2.444 at step
500 → 1.759 at 17000; standard-arm HumanEval 0/164 at every scored checkpoint — that
zero is the prompt-protocol artifact documented in
[`05_eval.md`](05_eval.md), not a capability zero). The 0922 end is read from the pod
log `runs/v41_gate_0922.log`: last val point `step 8000/38146 val 2.001`. Flat param
count was 3,209,392,128 total / 342,847,488 active (facts/v41.json
v41.gate_run_v41_gate_0911_summary config; S0 re-count
facts/corpus_supply.json#cs.gate_rebuild_m4_checklist_state_0921).

The pivot is recorded in `docs/standards/v41_pivot.md` (user order 2026-09-10, target
DeepSeek-V4.1-Flash). KDA/NoPE and AttnRes were retired with the earlier hybrid line;
the gate model carries position by partial RoPE and has no recurrent state.

## Why CED, and the abandoned A/B

`runs/prereg.jsonl#ced_vs_flat_0910@amended_1` registered a two-arm test: CED kept only
if matched-step val was at least as good as flat. User orders 2026-09-22/23 superseded
that design: flat stopped, CED runs single-arm, no control. The CED run therefore makes
no CED-vs-flat claim; the two runs also differ in data build (the six-domain rebalanced
mix and the rebuilt caches post-date 0911), so their losses are not comparable
(`runs/prereg.jsonl#v41_ced_0923@amended_1`, will_not_claim).

## CED structure

12 layers at d=1024, H=8. The bottom 6 layers (0–5) are the causal encoder; the top 6
(6–11) are the decoder. The mask is causal everywhere, so encoder layers are causally
identical to flat layers during the forward. The structural difference is where each
decoder layer gets its global KV:

- encoder layers 0–5 build learned non-overlapping 8-token global KV entries from their
  own hidden states (the `entries_per_doc` builder, `model.py:475`), exactly as the flat
  stack does;
- each decoder layer projects its OWN global KV entries and compression weights from
  the encoder boundary state H_6 through an unshared d→d `W_KV`/`W_Z` pair, mean-pooled
  per m-token document block before projection; no encoder layer carries that pair.

Attention in every layer is CSA2 (Compressed Sparse Attention 2): a lightweight indexer
selects top-k learned entries, then one concatenated softmax over
[selected global entries ; local SWA keys]. Layers 0–1 are SWA-only
(`--n_swa_only_layers 2`); every other layer also carries a local SWA branch, window
128. The entry branch is materialized and only the SWA window runs on flash-attn
(`--csa2_win_flash`): bf16-kernel-floor parity, 3.2–3.3× faster attention, 2.9× lighter
at T=4096 B4 (facts/v41.json#v41.de109_win_flash_parity_speed_0911). Position is
partial RoPE on the last 64 dims of each head (`--rope_dims 64`). No attention
residuals (`--no-attn_res`).

Every block is MoE: 48 experts, top-3 routed plus 1 shared, expert ffn 1728, fp32
router softmax with selection-only expert_bias, `torch._grouped_mm`
(`--moe_experts 48 --moe_top_k 3 --moe_shared 1 --moe_expert_ffn 1728 --moe_layers 0-11`).
Training is fp8 Float8Linear with `torch.compile`.

## Exact param count

| | total | active per token |
|---|---|---|
| CED gate stack | **3,221,975,040** | **355,430,400** |
| flat predecessor | 3,209,392,128 | 342,847,488 |

The CED count is the S0 meta-device exact count from `train.py --build_only` at the gate
shape, recorded in `runs/prereg.jsonl#v41_ced_0923@amended_1` question and the S2
experiment row. The +12.58M total / +12.58M active over flat is the six decoders'
unshared `W_KV`/`W_Z` projection pairs. The paper-scale design (40 layers, 20+20,
552B) is not built; every dimension was re-chosen for d=1024/seq 4096, with the
cross-check table in `docs/standards/v41_pivot.md`.

The shared-entry-builder fact is load-bearing for the later cleanup: the CED encoder
executes the flat code path, so removing flat is a branch refactor, not a deletion.
Evidence and the sequenced draft PRs live in
[`flat_refactor_sequence_0923.md`](flat_refactor_sequence_0923.md) and
[`non_ced_surface_analysis_0923.md`](non_ced_surface_analysis_0923.md), summarized in
[`07_next.md`](07_next.md).
