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

**These are quality domains, not instruction data, and the name of the intervention follows from that.** Checked at the builders rather than assumed: `cot` is `f"{problem}\n\n{solution}"` plain text (`datagen/numma_to_jsonl.py:34`, whose own docstring says "Plain problem+solution, not ChatML"); `textbook_30b` is the Phi-1 textbook-quality arm, also not instruction-shaped. Only `chatml` (rendered through `loader.format_example`) and `chat_qa` (the same rows in 问：/答： form) are instruction renderings, and together they are **0.114% of the mix**.

So this is not MiniCPM's intervention. Theirs mixes SFT-formatted data into the decay phase; ours would **raise the weight of quality domains during decay**, with the format leg absent. The mechanism transfers — concentrate quality where the learning rate collapses — and their +8 to +12 point magnitudes must not be quoted as an expectation for this.

The funding comes from `en_c4_stage2` and `zh_web` because those are the coarse-quality web half that MiniCPM's recipe *explicitly drops* from the decay phase — cutting them is the same move, not a side effect of needing tokens somewhere. Code and math give up 0.8× each, which is a trim rather than a cut: at 30B tokens they are the capability being trained and removing them from the last 10% would be a different experiment.

### Contamination: measured, and the zero means something narrower than it looks

A decay reweight concentrates whatever the raised domains contain into the phase with 19× the per-token weight, so the raised domains need a containment number, not an assumption. Against the current holdout set (29,923 items, fingerprint `26984613941ab5a2`), `scripts/holdout_containment.py` on the pod:

| domain | sampled | containment hits | rate |
|---|---|---|---|
| `chatml` | 2,668 | 0 | 0.0% |
| `chat_qa` | 2,668 | 0 | 0.0% |
| `cot` | 4,002 | 0 | 0.0% |
| `textbook_30b` | 4,002 | 0 | 0.0% |

Bounds, not zeros: rule of three puts each below 0.11% (2,668-row samples) or 0.075% (4,002-row) at 95%.

**The zero is "never had any", not "the filter removed them", and the difference is worth keeping even though the decision is the same.** None of the four has an `eval_contaminated` entry in its `build_corpus_stats.json` reasons histogram — `chatml` and `chat_qa` hold only `kept`, `cot` holds `kept`/`exact_dup`/`short`/`bad_bytes`. The pipeline is not demonstrated to have caught anything here; it had nothing to catch.

**A control had to be fixed before the zero could be read at all.** `holdout.load()` returns hashes, and feeding one back to `is_holdout()` returns False — a "positive control" built that way reports the guard broken while it is fine, which is a reversed conclusion shaped like a finding. The controls that stand take a real question string from a registry eval file (`control_sft_text_heldout` True, `humaneval_164` True) plus a negative control (plain Python source, False). The script now exits rather than printing a number if either fails.

### The constraint that shaped the numbers

Not supply — every domain has spare tokens. `cot` alone has 1.53B unused against a 420M ask. **The binding constraint is repeat exposure**, and the raises land it in a defensible place rather than at an arbitrary multiple:

- `cot` goes 0.44 → 1.38 pool-epochs. Below 1.0 the domain is not even seen once; the raise is the first time it is.
- `chatml` / `chat_qa` go 0.44 → ~1.56, still under 2.
- `textbook_30b` 1.89 → 2.03 is the largest absolute exposure and the one to argue about.
- Nothing exceeds its `epochs` cap in the file, so `build_mix` would not silently shrink a domain.

A 22.7× raise on `cot` sounds extreme and is not, because the base is 0.6% — the domain is nearly absent from the mix today. The exposure column is the honest scale, and by that measure the biggest change in the table is `chat_qa` going from unseen to seen 1.6 times.

## Two prerequisites, not suggestions

Neither depends on this proposal being accepted, and no number produced by a reweighted run is interpretable without both. 4c raised both from suggestion to precondition on 2026-09-08.

**1. The null.** Run the reweighting against the null that it does nothing.

**2. The seed-noise floor.** Run the *same* `anneal == weight` configuration twice, different seed. Without that magnitude, a later "+X%" cannot be told from X < noise — and this repo has already published a difference that turned out to be an instrument offset. The floor is the negative control for every anneal number that follows.

### The configuration, ready to launch

`data/mix_200m_4b.json` is the point to use, not a `mix_scale_*` ladder point: it carries **the same nine domains as the 30B v2 mix**, so a weight vector transfers without re-deriving anything. The ladder points carry a different seven-domain set (`web_hq`, `textbook`, `wiki`, `en`, `math`, `code`, `chat`) and nothing measured on them would name the domains this proposal moves.

Cost, from the run that completed: `p200m_4b_0902` did 3,814 steps / 4.0B tokens in **2h13m on 8 cards** (14:32→16:45, 2026-09-02). Three arms is roughly 6.7 card-hours × 8, and the two noise arms are independent of the third.

| arm | mix | `anneal_frac` | seed | purpose |
|---|---|---|---|---|
| N1 | `mix_200m_4b.json` unchanged | 0.10 | default | noise floor, run A |
| N2 | `mix_200m_4b.json` unchanged | 0.10 | default+1 | noise floor, run B — **N1 vs N2 is the measurement** |
| R | `mix_200m_4b.json` with this proposal's anneal column | 0.10 | default | the reweight |

`anneal_frac` must be **passed on the command line as 0.10 for all three**: the file declares `0.0`, and `_mix_anneal_frac` refuses a mix whose declared value disagrees with `Cfg` rather than silently picking one. At `0.0` there is no anneal phase at all and the whole design is inert.

Supply checked at this point, not assumed: applying the proposal's anneal weights to the 4B budget, every domain draws inside `supply × epochs`, and the highest exposure is 0.84 pool-epochs (`chatml`, `chat_qa`). Nothing repeats.

**Read N1 vs N2 before looking at R.** If |R − N1| is inside |N1 − N2|, the reweight did nothing measurable at this budget, and that is a result rather than a failed run.

## What this cannot say

Nothing here is measured on our models. The transfer is: MiniCPM at 1.1T tokens, dense Llama-style, SFT-formatted decay data. Ours would be 30B tokens, MoE with a looped stack, and four small curated domains, only 0.114% of which is instruction-formatted. **The mechanism transfers; the +8.2-point magnitude does not, and this document is not a prediction of a gain.**

Our own `−6.89%` anneal measurement (`docs/lessons/moe48_30b_0907_report.md`, 8B endpoint against its own `.step9000`, same rows and same mix, nine domains all falling) prices the learning-rate half alone. This proposal is the second half, and the two compose rather than compete.
