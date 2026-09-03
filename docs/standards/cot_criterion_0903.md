# cot fetch criterion (0903) — OpenThoughts-114k + Skywork-OR1-RL-Data

Written 2026-09-03, **before** reading any source row (aupai-6e: criterion before reading). The cot role in `mix_30b` targets 4.5B tokens of long CoT. Two open sources on hf-mirror: `open-thoughts/OpenThoughts-114k` (6 parquet, 1.08 GB) and `Skywork/Skywork-OR1-RL-Data` (4 parquet, 0.82 GB). Each is a mirror-chain fetch into a NEW dir (`data/corpus/cot_open_thoughts/` and `data/corpus/cot_skywork_or1/`), never a ladder-mix dir.

## Status of their fields
A source has these fields if it carries them; we tolerate absence per source:
- `input` / `domain` — the problem or code prompt.
- `reasoning` / `thinking` / `gt_cot` / `solution` / `answer` — the chain. On Skywork-OR1-RL the fields are RL-trajectory-shaped (`input`, `current`, `reflection`, `solution`, `adapted_solution`); on OpenThoughts they are `input`, `reasoning`, `response`. The chain is whichever field holds the multi-step reasoning.
- `response` / `final` — the answer.

## The keep criterion — a doc is KEPT iff ALL hold
1. **Has a multi-step chain.** The reasoning/thinking field exists and contains >1 distinct step (a sequence of deductions/edits), not a single token or an empty string.
2. **Chain is substantive (long CoT).** The chain is ≥ 200 chars of distinct text after whitespace-normalize. The role is *long* CoT; a one-line "answer = 42" with no chain is not this role.
3. **Complete.** The chain ends in a final answer/response; no truncated trace, no mid-generation kill, no `...` where the reasoning was cut off.
4. **Self-consistent for math.** When the problem is arithmetic/algebraic and the source provides a ground answer, the answer is derivable from the shown steps — the final value agrees with what the chain computes. A chain whose conclusion contradicts its own steps is dropped.
5. **Clean.** No interstitial author/sandbox artifacts (system-prompt leakage, reward-model ping, embedded training hash, `<|im_start|>`-style scaffold) inside the chain or response that the model would reproduce at inference.

## Pilot, per source (report BEFORE a full fetch)
Fetch one parquet slice per source, decode every row, apply the criterion, and report:
- pilot tokens per doc (frozen tokenizer) and total pilot tokens,
- reject rate and the reject histogram over the five checks,
- the host that served the slice (chain[0] hf-mirror unless it failed),
- measured reachable-to-target: pilot tokens × docs → extrapolated total vs the 4.5B target.

A reject that is majority cause #2 (not-long) or #4 (inconsistent math) is decided here, not carried into the build. A source that is 80%+ reagent-kept but lands far below 4.5B is reported as a supply shortfall, not padded.

## Stamping
`build_corpus_stats.json` carries this criterion's scalar outcome (docs_in, docs_kept, per-check rejects, tokens), the two sources' fingerprints, and `filters: cot-criterion-0903`. The controller stamps the domain into the mix only after reading the report.