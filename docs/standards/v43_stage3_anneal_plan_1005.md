---
question: After v42's 50,862-step 40B run, how do we structure the next stage to over-train on high-quality rewritten teaching code and finish on a quality anneal?
status: planned
source: controller order 2026-10-05 via awb mix#199157; method follows SmolLM2 (arXiv 2502.02737) and Qwen3 (arXiv 2505.09388)
---

# v43 — post-40B stage: high-quality over-training + final anneal

Plan, not result. Nothing here has run. Every token budget is an **estimate** for the
controller/jan to set; the only measured inputs are v42's committed mix
(`data/mix_v42_40b.json`, main 6aa44e21) and the cleaning dry-run below. Swallow-code's
gate-vocab token count is measured only after pack.

## What v42 already is (measured from the committed mix)

v42 40B is already code-heavy, not web-heavy:

| category | drawn tokens | share |
|---|---:|---:|
| code (l2 + l3 + starcoder) | 29.82B | 74.6% |
| math (owm_g4 + cot2) | 4.90B | 12.2% |
| zh_web | 3.10B | 7.7% |
| en_c4 | 2.00B | 5.0% |
| cot (CoT reasoning) | 0.18B | 0.4% |

Code pool capacity is 70.7B (l2 36.13B, l3 26.67B, starcoder 7.93B) against 29.82B drawn,
so v42 is not supply-bound on code rows. The binding fact is **quality distribution, not
code share**: the code is Ultra/starcoder at ~0.42 epochs, not pedagogically rewritten code.
The next stage therefore substitutes in teaching-grade code during over-training and
concentrates it in the anneal; it does not raise the code percentage.

## Reference schedules

- **Qwen3** (arXiv 2505.09388): S1 broad >30T; S2 ~5T (~14%) higher-quality with raised
  STEM/code/reasoning/synthetic share and **accelerated LR decay**; S3 long-context. The
  stage we are designing is the analogue of S2 — small budget, quality-concentrated, after
  the big general run.
- **SmolLM2-1.7B** (arXiv 2502.02737): three stable stages that progressively raise
  code/math (code 10% → 20% → more, math 0% → 5% → ~10%), then a **final anneal of 1T of
  11T = 9.1%**, mixture code 24% / math 14% / english-web 58% / cosmopedia 4%. Its base code
  share is 10%, so its 24% anneal-code is a large relative lift; our code share is already
  74.6%, so we keep anneal code share moderate and change *which* code (teaching-grade).

## The new ingredient

`tokyotech-llm/swallow-code` final config `ablation/exp11-scor/jsonl` = SwallowCode,
~16.1B upstream tokens of Python run through a four-stage pipeline (syntax validation,
pylint ≥7, SGCR style rewrite, SCOR self-contained educational rewrite) with
Llama-3.3-70B. This is the "rewritten teaching-grade code most useful for coding" the order
names; the ablation stages (exp2/5/10/...) are intermediate products and are **not** used.
License Llama-3.3 community + The-Stack-v2 terms.

Only exp11-scor is consumed. Sibling downloads (swallow-math, cosmopedia,
chinese-cosmopedia) are inputs to other owners' domains; this stage consumes them only if
the controller's final anneal calls for them.

Cleaning dry-run (2026-10-05, pod, 3,000 docs of the same-schema completed exp10 shard,
`build_corpus.py --filters light --dry`): kept 99.7% (near_dup 7, short 3). The SCOR output
is already clean, so the pipeline adds dedup + gate decontamination rather than filtering.

## Build pipeline (runs only after exp11-scor finishes downloading)

Code domain, frozen, new directory (never into an existing domain):

```bash
# 1) clean + exact dedup + MinHash near-dedup, domain-neutral filters (web filters mis-kill code)
python3 datagen/build_corpus.py --domain swallowcode_scor_dc \
  --source 'jsonl:data/hq_raw/tokyotech-llm_swallow-code/ablation/exp11-scor/jsonl/train-*.jsonl' \
  --filters light --workers <w> --phase v43
# 2) 13-word-token gate decontam (HumanEval + MATH-500 + GSM8K), write the _dc domain
python3 scripts/filter_gate_domains.py --domains swallowcode_scor \
  --out_names swallowcode_scor_dc --extra_math --workers 24
# 3) gate-vocab pack, CPU only (new domain, no v41/v42 domain touched)
CUDA_VISIBLE_DEVICES="" RAYON_NUM_THREADS=<t> python3 scripts/pretokenize_domains.py swallowcode_scor_dc --workers <w>
```

Constraints: no GPU (v41/v42 stages may be running); setsid+nohup detached; new domain name
so the v41 gate's frozen domains and v42's g4 domains are untouched; verify the triple
stamp (`f1f860970d15d623` / seed 42 / srcfp) before mix-writing. Step order matches the g4
delivery that already worked.

## Stage shape (estimates — controller/jan set the budgets)

WSD continuation reuses the stage-2 mechanism already in train.py: `--lr_origin_step
50862 --lr_peak_mult <f> --warmdown 1.0 --anneal_frac 0.0` re-warms from the v42 final
checkpoint over an absolute warmup then cosines to zero over the new segment.

- **Stage 3a — quality over-training.** Small relative budget (Qwen3 S2 analogue; order of
  ~10-15% of the 40B already spent = ~4-6B, **estimate**). Same category frame as v42 main
  but the code slot is a blend dominated by `swallowcode_scor_dc` for its highest-quality
  slice, no repeat (ep ≤ 1). LR re-warm to ~0.20-0.30 of peak (stage-2 used 0.30), then
  cosine down across the segment.
- **Stage 3b — final anneal.** ~10% of the stage-3 segment (SmolLM2 9.1% anchor =
  ~0.5-1.0B, **estimate**), LR cosines to zero. Quality-concentrated, teaching grade:
  initial target (**estimate**, tune after a domain-loss read on the HumanEval-distributed
  holdout): swallowcode_scor_dc ~40%, math reasoning (math_cot2_dc + swallow-math/finemath
  if those land) ~30%, teaching text (cosmopedia en + chinese-cosmopedia) ~20%, cot ~10%.
  Code share stays moderate because v42 is already 74.6% code; the lift is quality.

## Open decisions (not mine to set)

1. Stage-3 total token budget and the 3a/3b split (recipe = controller/user; jan launches
   after the current stage-2 is up).
2. Whether swallow-math and the two cosmopedia corpora join the anneal, and at what share —
   depends on their owners' finished, decontaminated domains.
3. SwallowCode is Python-only and English/Japanese-commented: accept the Python
   concentration (the gate is HumanEval Python) or cap its anneal share for diversity.
4. Mix weights derive from measured pools (as `write_mix_v42_40b.py` does); the v43 writer
   is written only after pack returns the real gate-vocab token count.

## Acceptance before this stops being a plan

- exp11-scor 8/8 shards present; `swallowcode_scor_dc` built with build_corpus_stats.json
  (`--filters light`, near_dedup true, extra_math_gates true, n=13) and kept/dropped
  fractions recorded as a fact.
- Gate-vocab pack returns measured tokens; the 16.1B upstream number is replaced by the
  measured value before any weight is set.
- Every v43 domain ≤ 1 epoch; triple stamps match; the v43 mix `--check`s against
  build_mix's per-domain floor exactly as the v42 writer does.
