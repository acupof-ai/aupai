---
question: What data does the V4.1 CED gate run train on, in what proportions, and where did each part come from?
status: measured
source: data/mix_v41_gate.json at the launch sha; facts/corpus_supply.json; facts/contamination.json; facts/tokenizer.json
---

# 01 — Gate mix, decontamination, tokenizer

The run consumes one mix: `data/mix_v41_gate.json`. `total_tokens` = 30.0B is the
budget, not the supply; per-domain `weight` is the main-phase target composition;
`anneal` is a separate composition for the last 10% of steps. Scheduling basis in
`train.py build_mix`: per-domain scheduled tokens = 30B × (0.9·weight + 0.1·anneal),
anneal_frac 0.10 (34,331 main steps + 3,815 anneal steps of 38,146).

## The six domains

| domain | weight w | anneal a | epochs | scheduled (B tok) | pool cap (B tok) | ratio sched/cap |
|---|---|---|---|---|---|---|
| code_ultra_l2_dc | 0.471951 | 0.316 | 1 | 13.691 | 36.161 | 0.379 |
| code_ultra_l3_noexec_dc | 0.314634 | 0.421333 | 1 | 9.759 | 26.699 | 0.366 |
| math_owm_stage2_dc | 0.080 | 0.100 | 1 | 2.460 | 9.465 | 0.260 |
| code_py_starcoder_dc | 0.073415 | 0.052667 | 1 | 2.140 | 7.949 | 0.269 |
| en_c4_stage2_dc | 0.045 | 0.040 | 1 | 1.335 | 5.784 | 0.231 |
| cot_dc | 0.015 | 0.070 | 3 | 0.615 | 1.200 | 0.513 |
| **sum** | **1.0** | **1.0** | | **30.000** | | |

Derived from the current `data/mix_v41_gate.json` (scheduled = arithmetic above) and its
per-domain `epochs_pool_source` full-load `measured_pool_tokens` (cap; cot ×3 epochs).
The earlier packed-token basis, measured 2026-09-11 on the pre-destruction caches, is
facts/corpus_supply.json#cs.gate_domains_dc_packed_0911 (six-domain total
58,230,309,922). The full-load pool figures are larger than the packed figures (L2
36.16B vs 15.34B): they are two counting bases for overlapping rows, and train.py's
epoch cap uses the full-load basis.

Supply facts for the current rebuild: L2 36,161,205,782 and L3 26,699,251,757
(facts/corpus_supply.json#cs.code_ultra_l2_dc_rebuilt_0921,
facts/corpus_supply.json#cs.code_ultra_l3_noexec_dc_rebuilt_0921); starcoder
7,948,956,977 and en_c4 5,783,727,841 exact document tokens
(facts/corpus_supply.json#cs.code_py_starcoder_dc_landed_0921,
facts/corpus_supply.json#cs.en_c4_stage2_dc_landed_0921). L3's 940 shards hold
15,419,384 rows after 109,551 13-gram drops and zero cross-group dups (mix file
`epochs_pool_source`).

No domain is scheduled above one pass except cot_dc (0.40B pool vs 0.615B scheduled,
hence 3 epochs). The tightest ratio is cot_dc at 0.513; under the 0911 packed basis L2
was tightest at 0.892, which no longer holds after the rebuild.

## Mix history

The 2026-09-10 pivot mix had eight domains. `code_keep_p1_dc` and `code_py_rp1t_dc`
were dropped by user order 2026-09-19 after the 2026-09-16 pod destruction: their raw
bytes and classifier labels were lost with no fingerprint-preserving source
(facts/corpus_supply.json#cs.gate_domains_dc_packed_0911 config.dropped_domains_history).
The freed 0.04 effective share returned to the three code domains pro rata; math, en_c4
and cot weights did not change. The domain set is kept only in the mix file.

## Decontamination

Every gate code domain is 13-word-gram decontaminated against HumanEval 164 and the MBPP
holdouts before it enters the mix. Two paths:

| path | domains | tool |
|---|---|---|
| non-ultra | starcoder, math_owm, en_c4, cot (and the dropped keep_p1/rp1t historically) | `python scripts/filter_gate_domains.py` over library `filters/decontam_ngram.py` |
| ultra | L2, L3-noexec | inside `datagen/ultradata_shards.py --aggregate`, before global dedup |

The 0911 non-ultra drop fractions were 0.0002%–0.00075% of rows, and a rescan found 0
residual hits (facts/contamination.json#cont.gate_dc_residual_0911; per-domain
cont.gate_dc_*). The current-pin audit re-ran the chain after the gate files moved and
concluded all six surviving domains are decontaminated against the current HumanEval/MBPP
pin, build stamp decontam_fp 302aa793f7067462
(facts/contamination.json#cont.gate_provenance_chain_0923). The same audit notes the
mix file's inline `fingerprint` field has no consumer aligning it to the loaded cache,
so 3 of 6 current builds have no cont.gate_dc_* fact of their own; the decontamination
stamp on the shards, not the mix field, is the operative identity.

L3 in the mix is the **no-exec** aggregate: static dedup, nontriviality floor and
decontam only, user order 2026-09-11. The execution-filtered L3 arm is retained outside
the mix for a later A/B.

## Tokenizer

| item | value | source |
|---|---|---|
| slots | 32,768 | facts/tokenizer.json#tok.vocab_size |
| specials | `<eos>=1`, `[NUM]=32767` | harness pinned_ids check |
| vocab fingerprint | f1f860970d15d623 | every gate cache `.srcfp` stamp |
| rebuild date / PR | 2026-09-10 / #233 | AGENTS.md tokenizer section |
| unfreeze condition | 2, corpus distribution changed | facts/tokenizer.json#tok.ultra_freeze_tax_0910 |
| measured freeze tax | ~8.5% more tokens per byte, old vocab on UltraData L2/L3, 3 document-disjoint seeds | facts/tokenizer.json#tok.ultra_freeze_tax_0910 |
| old vocab | 32,773 slots, pod-only `data/tokenizer_frozen_0829.json` | facts/tokenizer.json#tok.k6_vocab_fingerprint |

Every gate cache was rebuilt at the new vocab; a cache at an older vocab refuses to load.
Checkpoints and packs carry `vocab_id`, and a scoring run against the wrong vocab
refuses.
