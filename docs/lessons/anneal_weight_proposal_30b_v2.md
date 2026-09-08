---
question: What should the nine domains of mix_1.5b-a0.2b-e48_30b_v2 weigh during the anneal phase, given that MiniCPM's ablation says the decay-phase data choice is worth more than the SFT stage?
status: open
source: arXiv 2404.06395v3 Sec. 5 + Table 1; facts/data_scaling.json#ds.minicpm_decay_data_ablation; supply and epoch figures computed from data/mix_1.5b-a0.2b-e48_30b_v2.json, 2026-09-08
---

# An anneal weight proposal for the 30B v2 mix

**Proposal, not an edit.** The file is frozen by 4c's ruling of 2026-09-08 (its own `_comment`: v2 changes the architecture, so any weight change becomes a second variable and the prereg comparison stops being about the architecture). This document is the first version of a decision for whichever run is allowed to change data.

## What we do today, measured

Every domain of every mix in `data/` has `anneal == weight` to within 1e-9 — all 25 files, no exception. Our anneal phase changes the learning rate and leaves the data distribution byte-identical. We took MiniCPM's 10% length (`train.py`'s `Cfg.anneal_frac`, comment `(MiniCPM-style)`) and did not take the part their ablation is actually about.

Their result, in one line: doubling SFT tokens 6B → 12B moved C-Eval 40.9 → 41.2; relocating the same class of data into the decay phase moved it 40.9 → 49.1. The B-2 arm is the negative control and it is what makes the finding non-trivial — without it, A-1 vs A-2 could be read as "high-quality data helps", which nobody disputes.

**One caveat that has to travel with any use of this.** Their decay data is SFT-formatted, and our pretraining corpus contains effectively no ChatML — 0 occurrences of `<|im_start|>` in 168,000 rows across all 42 domains. Our `chatml` domain is 0.058% of the mix. So "mix SFT data into the decay phase" is not a thing we can do at their scale; what we have is four domains that are instruction-shaped or curated, and they are small.

## The proposal

`anneal_frac` stays 0.10. The anneal phase is 3.0B of the 30B budget. Only the `anneal` column changes.

| domain | weight (= anneal today) | proposed anneal | ×weight | anneal tokens | pool exposure now → proposed |
|---|---|---|---|---|---|
| `cot` | 0.006167 | **0.14** | 22.7× | 420M | 0.44 → 1.38 |
| `textbook_30b` | 0.101342 | **0.18** | 1.8× | 540M | 1.89 → 2.03 |
| `chatml` | 0.000577 | **0.015** | 26.0× | 45M | 0.44 → 1.55 |
| `chat_qa` | 0.000567 | **0.015** | 26.4× | 45M | 0.45 → 1.58 |
| `code_py_starcoder` | 0.391103 | 0.33 | 0.8× | 990M | 1.34 → 1.32 |
| `math_owm_stage2` | 0.288856 | 0.24 | 0.8× | 720M | 1.33 → 1.30 |
| `code_py_rp1t` | 0.018839 | 0.02 | 1.1× | 60M | 1.34 → 1.35 |
| `en_c4_stage2` | 0.162146 | **0.05** | 0.3× | 150M | 2.02 → 1.88 |
| `zh_web` | 0.030402 | **0.01** | 0.3× | 30M | 0.04 → 0.04 |

Sums to 1.0. The four raised domains take 35% of the anneal phase against 10.9% today.

### Why these four, and why funded from web

The four are the closest thing this corpus has to MiniCPM's "diverse and high-quality knowledge and ability-oriented" data: `cot` is chain-of-thought, `textbook_30b` is the Phi-1 textbook-quality arm, `chatml` and `chat_qa` are the same QA rows in two renderings.

The funding comes from `en_c4_stage2` and `zh_web` because those are the coarse-quality web half that MiniCPM's recipe *explicitly drops* from the decay phase — cutting them is the same move, not a side effect of needing tokens somewhere. Code and math give up 0.8× each, which is a trim rather than a cut: at 30B tokens they are the capability being trained and removing them from the last 10% would be a different experiment.

### The constraint that shaped the numbers

Not supply — every domain has spare tokens. `cot` alone has 1.53B unused against a 420M ask. **The binding constraint is repeat exposure**, and the raises land it in a defensible place rather than at an arbitrary multiple:

- `cot` goes 0.44 → 1.38 pool-epochs. Below 1.0 the domain is not even seen once; the raise is the first time it is.
- `chatml` / `chat_qa` go 0.44 → ~1.56, still under 2.
- `textbook_30b` 1.89 → 2.03 is the largest absolute exposure and the one to argue about.
- Nothing exceeds its `epochs` cap in the file, so `build_mix` would not silently shrink a domain.

A 22.7× raise on `cot` sounds extreme and is not, because the base is 0.6% — the domain is nearly absent from the mix today. The exposure column is the honest scale, and by that measure the biggest change in the table is `chat_qa` going from unseen to seen 1.6 times.

## What this cannot say

Nothing here is measured on our models. The transfer is: MiniCPM at 1.1T tokens, dense Llama-style, SFT-formatted decay data. Ours would be 30B tokens, MoE with a looped stack, and four small curated domains. **The mechanism (concentrating quality where the learning rate collapses) is what transfers; the +8.2-point magnitude does not.**

Our own `−6.89%` anneal measurement (`docs/lessons/moe48_30b_0907_report.md`, 8B endpoint against its own `.step9000`, same rows and same mix, nine domains all falling) prices the learning-rate half alone. This proposal is the second half, and the two compose rather than compete.

**The check worth running first is cheap and does not need this proposal.** Anything that reweights the anneal phase should be tested against the null that the reweighting does nothing, on a budget point small enough to run twice. The ladder mixes exist for exactly this.
