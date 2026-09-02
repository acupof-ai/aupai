---
question: What is the milestone-1 source list (rows, tokens, licence) for SFT corpus v1 — code-with-tests + math CoT, 200M tokens, ChatML, packed by prepare_sft_math (FoNE off)?
status: measured
source: read-only survey subagent over /Users/bytedance/code/aupai-3b repo + /work/aupai pod, 2026-09-02
---

# SFT corpus v1 — milestone 1 source list (3b-6)

## Key finding

The **code-with-tests** cell (the one genuine gap from the 30B code-supply ruling) has **no
materialized SFT source on the pod**. `data/sft/fable5_cot.jsonl` is not code-with-tests —
353 rows / 1.43 MB of general CoT (`{prompt,response,category}`). The ruling's code-with-tests
supply must be *synthesized* by the Fable generator + sandbox verifier, a pipe not yet built.
It cannot be met from any fetched dataset.

## Candidate sources (verified on pod, /work/aupai/data)

| # | Source | Rows | Est. tokens | Licence | Language | Note |
|---|---|---|---|---|---|---|
| 1 | raw/hf_numma/*.parquet (NuminaMath-CoT-1.5) | 859,494 | ~690M | Apache-2.0 | EN math CoT | aops, amc_aime, synthetic_math |
| 2 | raw/ms_om2/*.parquet (OpenMathInstruct-2) | 21,973,791 | ≥35M | MIT — verify | EN math CoT | 32 shards; ms_om2 valid |
| 3 | raw/hf_finemath_4plus/*.parquet | 6,699,493 | ~1000 | Apache-2.0 | EN math text | document-style, not Q→A |
| 4 | school_math_r1_zh.jsonl | 223,423 | ~54M | research — verify | ZH math CoT | R1-distilled |
| 5 | math/belle.jsonl | 236,940 | ~66M | MIT | ZH math | |
| 6 | math/mxode.jsonl | 211,988 | ~53M | flag | ZH math | verbose step-by-step |
| 7 | math/ape210k.jsonl | 192,764 | ~44M | MIT | ZH math | answer noise (`\boxed`) |
| 8 | math/math23k.jsonl | 20,026 | small | | ZH math | high containment 74% |
| 9 | math/gsm8k_zh.jsonl + workbatch/gsm8k_zh_train | 7,471 | ~1M | MIT; zh verify | ZH | `####`→`答案是：` |
| 10 | sft/fable5_cot.jsonl | **353** | ~0.35M | unverified — flag | EN/zh | CoT, **not code-with-tests** |
| 11 | raw/fable5_2m/data/train.parquet (full Fable-5) | 2,006,487 | ~900 | unverified — flag | EN/zh | needs cleaning to SFT |
| 12 | sft_math_code.jsonl | 15,471 | ~4.6M | internal synth | ZH code+math | per-row source tag |
| 13 | synthetic_code.jsonl | 3,000 | ~0.3M | internal synth | ZH code | python with run-output |
| 14 | raw/code_supply, raw/ms_starcoder_py (The Stack) | code files | large | per-file multi | EN code | no test pairs |

## Recommended 200M-token mix (materialized sources only)

| Source | Tokens | Rows to pack | Share |
|---|---|---|---|
| NuminaMath-CoT | 90M | 112,500 | 45% |
| OpenMathInstruct-2 | 35M | 70,000 | 17.5% |
| Finemath-4-Plus | 20M | sample | 10% |
| zh math (school_math_r1+belle+ape210k+mxode+math23k+gsm8k) | 30M | ~125k | 15% |
| Fable-5 CoT | 10M | ~10k | 5% |
| code (sft_math_code+synthetic_code) | 10M | ~30k | 5% |
| general replay (alpaca/coig/openo1) | 5M | ~15k | 2.5% |
| **Total** | **200M** | | |

## Hazards

- **Code-with-tests shortfall ~2×**: ~5M code tokens available vs 10M target. Requires the
  Fable generator+sandbox pipe (or cleaning fable5_2m) before v1 hits the intended split.
- **Fable undersupplied ~30×**: fable5_cot is 353 rows. Same pipe requirement.
- **Contamination**: packer holdout (datagen/holdout.py EVAL_FILES) excludes only
  math_test_500 / math_hard_eval_1k / code_holdout_500 / code_holdout_v2_500. math-500 and
  the MC suite (math-500 saturated) are NOT excluded by the packer. NuminaMath + OpenMathInstruct
  contain GSM8K-family + AMC/AIME problems overlapping math-500 / math-hard — scan before pack.
- **Licence**: verify OpenMathInstruct-2 (MIT card), Fable (not recorded), The-Stack (per-file),
  COIG (cc-by-nc research-restricted), alpaca_gpt4_zh (GPT-4-derived, murky), school_math_r1
  (distillation provenance).