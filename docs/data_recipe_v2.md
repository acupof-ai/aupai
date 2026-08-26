# Pretraining Data Recipe v2 (1.27B tokens)

## Sources

| Source | Chars | Tokens (est.) | Share | Type |
|--------|-------|---------------|-------|------|
| minimind pretrain_hq | 525M | ~350M | 28% | Chinese conversational |
| Chinese Cosmopedia (textbook+wikihow) | 642M | ~428M | 34% | Chinese textbook |
| Existing corpus (pretrain_tagged) | 203M | ~136M | 11% | Chinese mixed |
| English Cosmopedia (auto_math_text + openstax) | 420M | ~280M | 22% | English textbook/math |
| Evol-instruction-66k | 114M | ~76M | 6% | English code instruction |
| **Total** | **1.91B** | **~1.27B** | | |

## Rationale

- Target recipe (from research): 55% ZH / 25% EN textbook / 12% code / 8% math
- Actual: 73% ZH / 22% EN / 6% code (code underrepresented, acceptable per "quality > ratio" finding)
- tok/param: 1.27B / 200M = 6.35 (was 0.68, Chinchilla optimal = 20)
- 2 epochs recommended (SmolLM2, phi-4 evidence)

## Key research findings applied

1. Quality filtering > ratio optimization at this scale (DCLM)
2. 2-3 epochs over filtered subset beats 1 epoch of superset (2503.07879)
3. No curriculum needed at <1B tokens; quality-anneal at end instead
4. Long-CoT only in SFT, not pretraining (Through the Valley, 2506.07712)
5. Per-dataset epoch cap: 4-5 (SmolLM2 degradation threshold)

## Files

- `/work/aupai/data/pretrain_1b.jsonl` — final mixed dataset (1.66M docs)
- Source data cached on pod under `/work/aupai/data/`
